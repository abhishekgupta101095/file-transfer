from bs4 import BeautifulSoup
import urllib.parse
import lxml
import os
import re
import boto3
import json
from botocore.config import Config
from pathlib import Path
import time
from wand.image import Image
import docx2text
import xlsx2text
from dynamodblog import table_log, table_textract
import hashlib
from pypdf import PdfReader, PdfWriter


#runtime_region = os.environ('AWS_REGION')

textract_absent_region=["ap-northeast-1"]
comprehendmedical_absent_region=["ap-northeast-1", "ap-southeast-1"]


# COMPREHEND_CLIENT_INSURANCE = boto3.client(
#     "comprehend", region_name="us-east-2")


COMPREHEND_CLIENT = boto3.client(
    "comprehend",
    config=Config(retries={"max_attempts": 20, "mode": "standard"}, connect_timeout=30),
)
#if runtime_region in comprehendmedical_absent_region:
    
COMPREHENDMEDICAL_CLIENT = boto3.client(
        "comprehendmedical", 
        config=Config(retries={"max_attempts": 20, "mode": "standard"}, connect_timeout=30),
    )
# else:
#     COMPREHENDMEDICAL_CLIENT = boto3.client(
#         "comprehendmedical", 
#         config=Config(retries={"max_attempts": 20, "mode": "standard"}, connect_timeout=30),
#     )
    
S3_CLIENT = boto3.client("s3")
S3_RESOURCE = boto3.resource("s3")

#if runtime_region in textract_absent_region:
TEXTRACT_CLIENT = boto3.client(
        "textract", 
        config=Config(retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=30),
    )
# else:
#     TEXTRACT_CLIENT = boto3.client(
#         "textract", 
#         config=Config(retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=30),
#     )
    

DYNAMODB_RESOURCE = boto3.resource(
    "dynamodb",
    config=Config(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=30),
)


# service_type_mapping = {"NB": "New Business", "REN": "Renewal", "END": "Endorsement"}
service_type_mapping = {"NB": "New Business", "REN": "Renewal", "END": "Endorsement", "Password":"Password email", 
"RFP":"RFP email", "RFI":"RFI email"}

def aws_insurance_model_init():
    aws_insurance_s3_source = "pncdidev-cip-inference-store-models-241231414837-us-east-2"
    source_key = "pncattachmenttypeNC/insurancemodel/service-2.json"

    file_name = source_key.split("/")[2]

    aws_insurance_s3_dest = "/tmp/botodata/"

    if not os.path.exists(aws_insurance_s3_dest):
        os.mkdir(aws_insurance_s3_dest)

    try:
        S3_CLIENT.download_file(aws_insurance_s3_source, source_key, os.path.join(aws_insurance_s3_dest,file_name))
        os.environ["AWS_DATA_PATH"] =  aws_insurance_s3_dest
        print("file upload successful", os.path.join(aws_insurance_s3_dest,file_name))
    except Exception as error:
        print(error)
    

def archive_file(copy_source=None, dest_bucket=None, dest_filename=None):
    """
    This function will move file between s3 locations
    """
    # Move the email to its correct place. If email is skipped then move to a "skipped" prefix
    try:
        copy_dest = S3_RESOURCE.Bucket(dest_bucket)
        copy_dest.copy(copy_source, dest_filename)
        S3_CLIENT.delete_object(Bucket=copy_source["Bucket"], Key=copy_source["Key"])
        print(f"File moved to {dest_bucket}/{dest_filename}")
    except Exception as error:
        print(error)
        print(
            f"Error moving file from {json.dumps(copy_source)} to {dest_bucket}/{dest_filename}"
        )


def upload_to_s3_asdata(content, bucket, key):
    """
    This function will update memory data into s3
    """
    try:
        if isinstance(content, dict):
            content = str(json.dumps(content))
        S3_CLIENT.put_object(Body=bytes(content, "utf-8"), Bucket=bucket, Key=str(key))
        print(f"Upload successful to {bucket}/{key}")
    except Exception as error:
        print(f"Issue with upload to {bucket}/{key}")
        raise error


class Classifier:
    def __init__(self, endpoint, request_type):
        self.endpoint = endpoint
        self.request_type = request_type

    def split_s3_path(self, s3_path):
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        return bucket, key

    def extract_text_from_file(self, filename, hash_digest):
        text = ""
        if ".xlsx" in filename:
            print("file extention changed to xlsx to avoid timeouts")
            filename.suffix = ".xlsx"
            
        if filename.suffix in [".docx"]:
            text = docx2text.process(filename)
        elif filename.suffix in [".xlsx"]:
            text = xlsx2text.process(filename)
        elif filename.suffix in [".csv"]:
            #            with open(filename, "r") as csvfile:
            with open(filename, "r", encoding="utf-8", errors="ignore") as csvfile:
                text = csvfile.read()
        # ASP
        elif filename.suffix in [".txt"]:
            with open(filename, "r") as txtfile:
                text = txtfile.read()
        elif filename.suffix in [".html"]:
            with open(filename, "r") as htmlfile:
                text = htmlfile.read()
        if text:
            text = text.replace("\n", " ")
            text = re.sub(r"\s\s+", " ", text)
            return text
        else:
            textract_bucket = os.getenv("TEXTRACT_BUCKET")
            textract_table = os.getenv("TEXTRACT_TABLE")
            textract_table = DYNAMODB_RESOURCE.Table(textract_table)

            table_data = table_textract(textract_table, "get", hash_digest, "")
            raw_textract = dict()

            if table_data is not None:
                txt_output = table_data["textractoutput"]
                print(
                    f"Textract data available for filehash {hash_digest} in {txt_output}"
                )
                txt_bucket, txt_key = self.split_s3_path(txt_output)
                contents = S3_CLIENT.get_object(Bucket=txt_bucket, Key=txt_key)
                raw_textract = contents["Body"].read()
                raw_textract = json.loads(raw_textract)
                table_textract(
                    textract_table,
                    "update",
                    hash_digest,
                    {},
                    "accesscount",
                    int(table_data.get("accesscount", 0)) + 1,
                )
            else:
                print(f"Textract data NOT available for filehash {hash_digest}")
                temp_filename = Path(Path(filename).parent).joinpath("temp.png")

                with Image(filename=filename, resolution=300) as pdf:
                    for count, page in enumerate(pdf.sequence):
                        with Image(page) as img:
                            img.format = "png"
                            img.save(filename=temp_filename)
                        with open(temp_filename, "rb") as img:
                            img_data = img.read()
                            raw_response = TEXTRACT_CLIENT.detect_document_text(
                                Document={"Bytes": img_data},
                            )
                            # raw_response = TEXTRACT_CLIENT.analyze_document(
                            #     Document={"Bytes": img_data},
                            #     FeatureTypes=["FORMS", "TABLES"],
                            # )
                            raw_textract[str(count + 1)] = raw_response

                        Path.unlink(temp_filename)

                upload_to_s3_asdata(
                    raw_textract,
                    textract_bucket,
                    f"{hash_digest}.json",
                )
                print(f"Uploaded Textract data to {textract_bucket}/{hash_digest}.json")
                table_textract(
                    textract_table,
                    "insert",
                    hash_digest,
                    {
                        "textractoutput": f"s3://{textract_bucket}/{hash_digest}.json",
                        "filename": str(Path(filename).name),
                        "accesscount": 0,
                    },
                )
                print(f"Inserted Textract record for filehash {hash_digest}")

            textract_contents = []
            for _, page_response in raw_textract.items():
                try:
                    # Added logic to identify right template in case of sunlife
                    # RSP was being identified as RCPP
                    minimum_weight = 100
                    maximum_weight = 0
                    for block in page_response["Blocks"]:
                        if block["BlockType"] in ["LINE"]:
                            weight = float(block["Geometry"]["BoundingBox"]["Height"])
                            minimum_weight = min(minimum_weight, weight)
                            maximum_weight = max(maximum_weight, weight)
                            
                    weighted_text = []
                    
                    for block in page_response["Blocks"]:
                        if block["BlockType"] in ["LINE"]:
                            multiplier = int(weight / minimum_weight)* 10
                            for _ in range(multiplier):
                                weighted_text.append(block["Text"].lower())
                    
                    response = weighted_text
# commented below logic and got the code in line with ocr-extract for the RSPP fix                        
                    # response = [
                    #     block["Text"].lower()
                    #     for block in page_response["Blocks"]
                    #     if block["BlockType"] in ["LINE"]
                    # ]
                except Exception as error:
                    print(f"Error after for {error}")
                    response = []
                textract_contents += response

            return " ".join(textract_contents)

    def service_type_classifier(self, data, submission_id, table):
        print(f"Starting {self.request_type} inference")
        self.classification_type = ""
        if data:
            servicetype_threshhold = float(os.getenv("SERVICETYPE_THRESHHOLD"))
            soup = BeautifulSoup(data, features="lxml")
            data = soup.get_text()
            data = data.split("________________________________")[0]
            data = data.replace("\n", " ").replace("\r", " ")
            data = data[:4990]
            print(f"AWS Comprehend Request : {data}")
            try:
                response = COMPREHEND_CLIENT.classify_document(
                    Text=data, EndpointArn=self.endpoint
                )
                print("AWS Comprehend Response : ", response["Classes"])
                top_class = response["Classes"][0]
                self.classification_type = top_class["Name"]
                if (
                    self.classification_type.lower() == "negative"
                    or top_class["Score"] < servicetype_threshhold
                ):
                    self.classification_type = ""
                    print("Unable to identify the TYPE of request")
                else:
                    print(
                        f"Text has been classified with Service Type : {self.classification_type}"
                    )
            except Exception as error:
                print(error)
        else:
            print("No Text found to pass to Comprehend. Request Type not tagged.")

        self.classification_type = service_type_mapping.get(
            self.classification_type, ""
        )

        return self.classification_type

    # def attachment_type_classifier(self, data, submission_id, table):
    #     print(f"Starting {self.request_type} inference")
    #     self.classification_type = []

    #     attachmenttype_minimum_text = int(os.getenv("ATTACHMENTTYPE_MINIMUM_TEXT"))

    #     if data:
    #         table_log(
    #             table,
    #             "update",
    #             submission_id,
    #             f"documents",
    #             value=data,
    #         )
    #         table_log(
    #             table,
    #             "update",
    #             submission_id,
    #             f"countofdocuments",
    #             value=len(data),
    #         )
    #         for count, attach in enumerate(data):
    #             attach = urllib.parse.unquote_plus(attach, encoding="utf-8")

    #             table_log(table, "insert", f"{submission_id}-{count+1}", "starttime")
    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{count+1}",
    #                 "documents",
    #                 value=attach,
    #             )

    #             try:
    #                 bucket, key = self.split_s3_path(attach)
    #             except Exception as error:
    #                 print(f"Invalid attachment name {attach}")
    #                 raise error

    #             try:
    #                 data = S3_CLIENT.get_object(Bucket=bucket, Key=key)
    #                 contents = data["Body"].read()  # .decode("utf-8", "ignore")
    #             except Exception as error:
    #                 print(
    #                     "Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.".format(
    #                         key, bucket
    #                     )
    #                 )
    #                 raise error

    #             print(f"Downloaded {attach}")

    #             file_name = Path("/tmp").joinpath(Path(key).name)
    #             with open(file_name, "wb") as download_file:
    #                 download_file.write(contents)
    #                 hash_digest = hashlib.sha256(contents).hexdigest()

    #             extracted_text = ""

    #             try:
    #                 extracted_text = self.extract_text_from_file(file_name, hash_digest)
    #                 # An error occurred (UnsupportedDocumentException) when calling the AnalyzeDocument operation: Request has unsupported document format
    #                 # response = TEXTRACT_CLIENT.analyze_document(
    #                 #     Document={"S3Object": {"Bucket": bucket, "Name": key}},
    #                 #     FeatureTypes=["FORMS"],
    #                 # )
    #                 # print(response)
    #             except Exception as error:
    #                 print(error)

    #             extracted_text = extracted_text.encode("ascii", "ignore")
    #             extracted_text = extracted_text.decode()

    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{count+1}",
    #                 "textract",
    #                 value=extracted_text,
    #             )

    #             extracted_text = extracted_text[:4990]

    #             if len(extracted_text) >= attachmenttype_minimum_text:
    #                 print(f"AWS Comprehend Request : {extracted_text}")
    #                 try:
    #                     for i in range(3):
    #                         try:
    #                             response = COMPREHEND_CLIENT.classify_document(
    #                                 Text=extracted_text, EndpointArn=self.endpoint
    #                             )
    #                             print("AWS Comprehend Response : ", response["Classes"])
    #                         except Exception as error:
    #                             print(error)
    #                             print(f"Trying #{i+1}")
    #                             time.sleep(60)
    #                             continue
    #                         break
    #                     else:
    #                         raise Exception("Maximum retries exhausted")

    #                     table_log(
    #                         table,
    #                         "update",
    #                         f"{submission_id}-{count+1}",
    #                         "result",
    #                         value=json.dumps(response["Classes"]),
    #                     )
    #                     top_class = response["Classes"][0]

    #                     self.classification_type.append(top_class)
    #                 except Exception as error:
    #                     print(error)
    #                     self.classification_type.append({"Name": "OTHER", "Score": 1})
    #             else:
    #                 print(
    #                     f"Document Text Length: {len(extracted_text)}. Text in document is less than {attachmenttype_minimum_text} characters. Request Type tagged as NONDOCUMENT."
    #                 )
    #                 self.classification_type.append({"Name": "NONDOCUMENT", "Score": 1})

    #             self.classification_type[-1]["filehash"] = hash_digest

    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{count+1}",
    #                 "result",
    #                 value=json.dumps(self.classification_type[-1]),
    #             )

    #             Path.unlink(file_name)

    #             table_log(
    #                 table, "update", f"{submission_id}-{count+1}", "completiontime"
    #             )

    #     return self.classification_type

    def claimsattachment_type_classifier(
        self,
        data,
        submission_id,
        table,
        trimmed_output,
        is_document_translated,
        language,
    ):
        print(f"Starting {self.request_type} inference")
        self.classification_type_details = []

        attachmenttype_minimum_text = int(os.getenv("ATTACHMENTTYPE_MINIMUM_TEXT"))
        comprehend_json_suffix = os.getenv("COMPREHEND_JSON_SUFFIX")
        gen_ai_landing_bucket = os.getenv("GEN_AI_LANDING_BUCKET")

        if data:
            table_log(
                table,
                "update",
                submission_id,
                f"documents",
                value=data,
            )
            table_log(
                table,
                "update",
                submission_id,
                f"countofdocuments",
                value=len(data),
            )

            for filepath in data:
                if not isinstance(filepath, dict):
                    filepath = json.loads(filepath)

                attach = filepath["path"]
                doc_id = filepath["document_id"]
                table_log(table, "insert", f"{submission_id}-{doc_id}", "starttime")
                table_log(
                    table,
                    "update",
                    f"{submission_id}-{doc_id}",
                    "documents",
                    value=attach,
                )
                table_log(
                    table,
                    "update",
                    f"{submission_id}-{doc_id}",
                    "model",
                    value=self.request_type,
                )
                old_attach = attach
                if is_document_translated == "Y":
                    attach = f"{attach}.translate"
                    bucket, key = self.split_s3_path(attach)
                    translate_filename = key
                    print("Processing translated document")
                else:
                    print("Processing non-translated document")
                    
                try:
                        bucket, key = self.split_s3_path(attach)
                except Exception as error:
                        print(f"Invalid attachment name {attach}: Error: {error}")
                        continue
                   

                try:
                    data = S3_CLIENT.get_object(Bucket=bucket, Key=key)
                    contents = data["Body"].read()  # .decode("utf-8", "ignore")
                    hash_digest = hashlib.sha256(contents).hexdigest()
                    print(f"file hash is - {hash_digest}")
                except Exception as error:
                    print(
                        "Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.".format(
                            key, bucket
                        )
                    )
                    print(error)
                    continue

                print(f"Downloaded {attach}")
                
                # aws_insurance_model_init()
                
                if is_document_translated == "Y":
                    print("in if")
                    extracted_text = contents.decode("utf-8", "ignore")
                    trans_extracted_text = extracted_text
                    try:
                        bucket, key = self.split_s3_path(old_attach)
                    except Exception as error:
                        print(f"Invalid attachment name {attach}: Error: {error}")
                        continue
                    try:
                        data = S3_CLIENT.get_object(Bucket=bucket, Key=key)
                        contents = data["Body"].read()
                    except Exception as error:
                        print(
                            "Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.".format(
                                key, bucket
                            )
                        )
                        print(error)
                        continue
                    file_name = Path("/tmp").joinpath(Path(key).name)
                    print("Filename after joining:",file_name)
                    with open(file_name, "wb") as download_file:
                        download_file.write(contents)
                    #print("Contents are:",contents)
                    try:
                        temp_filename = Path(Path(file_name).parent).joinpath("temp.png")
                        with Image(filename=file_name, resolution=100) as pdf:
                            for count, page in enumerate(pdf.sequence):
                                if count == 0:
                                    with Image(page) as img:
                                        img.format = "png"
                                        img.save(filename=temp_filename)
                                    break    
                        tmp_key = f"{key}.png"                                
                        # print(f"var before upload are - {str(temp_filename)}, {bucket}, {tmp_key}")                                
                        S3_CLIENT.upload_file(str(temp_filename), Bucket=bucket, Key=str(tmp_key))
                        print(f"temp_file saved in {bucket}, {tmp_key}")
                        with open(temp_filename, "rb") as img:
                            img_data_new = img.read()
                            # print ("image_data_new output is ", img_data_new)
                    #         response = COMPREHEND_CLIENT.classify_document(
                    #                 EndpointArn=self.endpoint,
                    #                 Bytes=img_data_new,
                    #                 DocumentReaderConfig={
                    #                     'DocumentReadAction': 'TEXTRACT_ANALYZE_DOCUMENT',
                    #                     'DocumentReadMode': 'FORCE_DOCUMENT_READ_ACTION',
                    #                     'FeatureTypes': [
                    #                         'TABLES','FORMS'
                    #                                     ]
                    #                                     }
                    #                                               )
                    #         print("AWS Comprehend Response : ", response["Classes"])  
                    except Exception as error:
                        print("Classification error after translation",error)
                else:
                    print(f"Inside else at line 540")
                    print("checking abcs")
                    file_name = Path("/tmp").joinpath(Path(key).name)
                    try:
                        with open(file_name, "wb") as download_file:
                            # print(f"contents for classifcation are - {contents}")
                            download_file.write(contents)
                            # hash_digest = hashlib.sha256(contents).hexdigest()
                            # print(f"hash of the file is - {hash_digest}")
                    except Exception as error:
                        print(f"inside error {error}")
                    extracted_text = ""

                    try:
                        print("in try 590")
                        temp_filename = Path(Path(file_name).parent).joinpath("temp.png")
                        # with Image(filename=file_name, resolution=300) as pdf:
                        #     for count, page in enumerate(pdf.sequence):
                        #         if count == 0:
                        #             with Image(page) as img:
                        #                 img.format = "png"
                        #                 img.save(filename=temp_filename)
                        #             break  
                        pdf_file = PdfReader(file_name)
                        print("try 600")    
                        if len(pdf_file.pages) > 0:
                            first_page = pdf_file.pages[0]
                            pdf_writer = PdfWriter()
                            pdf_writer.add_page(first_page)
                            with open(temp_filename, "wb") as one_page_pdf:
                                pdf_writer.write(one_page_pdf)
                                print(f"temp_fiiename written successfully")
                        else:
                            print("the pdf file is empty")
                        print("try 610")    
                        tmp_key = f"{key}_classification.pdf"
                        # print(f"var before upload are - {str(temp_filename)}, {bucket}, {tmp_key}")
                        S3_CLIENT.upload_file(str(temp_filename), Bucket=bucket, Key=str(tmp_key))
                        print(f"temp_file saved in {bucket}, {tmp_key}")

                        with open(temp_filename, "rb") as img:
                            print("try 617")
                            img_data_new = img.read()
                        print("try 618")
                        extracted_text = self.extract_text_from_file(
                            file_name, hash_digest
                        )
                        print("extracted_text 620",extracted_text)
# Changes made by Ayushi and Kavitha for SunLife on 25th Aug 2023
                        table_log(
                            table,
                            "update",
                            f"{submission_id}-{doc_id}",
                            "filehash",
                            value=hash_digest,
                        )
                        print("try632")
                    except Exception as error:
                        print("except634")
                        print(error)
                    print("line 636")
                print("extracted_text 630",extracted_text)
                if is_document_translated == "Y":
                    extracted_text = trans_extracted_text
                print("extracted_text 633",extracted_text)
                extracted_text = extracted_text.encode("ascii", "ignore")
                extracted_text = extracted_text.decode()
                print("extracted_text 633",extracted_text)

                table_log(
                    table,
                    "update",
                    f"{submission_id}-{doc_id}",
                    "textract",
# ASP fixed the issue pertaining to size exceed in inferencelog table                    
#                    value=extracted_text,
                    value="",
                )

                extracted_text_pii = extracted_text[:99990]
                extracted_text_phi = extracted_text[:19990]
                extracted_text = extracted_text[:4990]
                print("extracted_text_pii 647",extracted_text_pii)
                print("extracted_text_phi",extracted_text_phi)
                print("extracted_text",extracted_text)
                result = {
                    "object_link": old_attach,
                    "language": language,
                    "document_id": doc_id,
                }
#                 Arn = "arn:aws:comprehend:us-east-2:aws:document-classifier-endpoint/insurance"
#                 try:
# # For comprehend insurance API call 
#                     print("starting new comprehend insurance call")
#                     response = COMPREHEND_CLIENT_INSURANCE.classify_document(
#                                 Bytes=img_data_new, EndpointArn= Arn
#                                 )
#                     print("AWS Comprehend Insurance Response : ", response)
#                 except Exception as error:
#                     print(error)

                try:
                    for i in range(3):
                        try:
                            print("Started comprehend call")
                            response = COMPREHEND_CLIENT.classify_document(
                            EndpointArn=self.endpoint,
                            Bytes=img_data_new,
                            DocumentReaderConfig={
                                'DocumentReadAction': 'TEXTRACT_ANALYZE_DOCUMENT',
                                'DocumentReadMode': 'FORCE_DOCUMENT_READ_ACTION',
                                'FeatureTypes': [
                                    'TABLES','FORMS'
                                        ]
                                    }
                                                                  )
                            print("AWS Comprehend Response : ", response["Classes"])
                            metadata = {key: str(value) for key, value in response["Classes"][0].items()}
                            print(f"metadata - {metadata} ")

    # Below copy_object command is ccopying the currents attachment for classification to the gen-ai-destination bucket
    # with metadata updated with the document classification type with highest confidence. 
                            try:
                                S3_CLIENT.copy_object(Key=key, Bucket=gen_ai_landing_bucket,
                                        CopySource={"Bucket": bucket, "Key": key},
                                        Metadata=metadata,
                                        MetadataDirective="REPLACE")
                                print(f"data copy successful")        
                                

                            except Exception as error:
                                print("Error while copying metadata", error)
                                        
                        except Exception as error:
                            print(error)
                            print(f"Trying #{i+1}")
                            time.sleep(60)
                            continue
                        break
                    else:
                        raise Exception("Maximum retries exhausted")
                        
                    result["Classes"] = response["Classes"]
                    if self.request_type == "gbattachmenttype" and "emailbody" in key:
                        result["Classes"][0] = {
                                "Name": "emailbody",
                                "Score": 1.0,
                                "Page": 1
                            }
                        
    
                        
                except Exception as error:
                    print(error)
                    result["Classes"] = [
                            {
                                "Name": "CLASSIFICATION_FAILED",
                                "Score": 1.0,
                            }
                        ]
                
                
                if len(extracted_text) >= attachmenttype_minimum_text:
#ASP commenting below lines as the comprehend call shifted above this if.                     
#                     print(f"AWS Comprehend Request : {extracted_text}")

#                     try:
#                         for i in range(3):
#                             try:
#                                 print("Entered actual classification call")
#                                 # response = COMPREHEND_CLIENT.classify_document(
#                                 #     Text=extracted_text, EndpointArn=self.endpoint
#                                 # )
# # Changes made by Ayushi and Kavitha for SunLife on 25th Aug 2023  
#                                 response = COMPREHEND_CLIENT.classify_document(
#                                     EndpointArn=self.endpoint,
#                                     Bytes=img_data_new,
#                                     DocumentReaderConfig={
#                                         'DocumentReadAction': 'TEXTRACT_ANALYZE_DOCUMENT',
#                                         'DocumentReadMode': 'FORCE_DOCUMENT_READ_ACTION',
#                                         'FeatureTypes': [
#                                             'TABLES','FORMS'
#                                                         ]
#                                                         }
#                                                                   )
#                                 print("AWS Comprehend Response : ", response["Classes"])
#                             except Exception as error:
#                                 print(error)
#                                 print(f"Trying #{i+1}")
#                                 time.sleep(60)
#                                 continue
#                             break
#                         else:
#                             raise Exception("Maximum retries exhausted")

#                         result["Classes"] = response["Classes"]
                        if trimmed_output == "Y":
                            print(
                                "PHI and PII data will be trimmed from output and no output json will be created"
                            )
                            result["piidata"] = {}
                            result["phidata"] = {}
                        else:
                            response = COMPREHEND_CLIENT.detect_pii_entities(
                                Text=extracted_text_pii, LanguageCode="en"
                            )

                            print("AWS Comprehend PII Response : ", response)

                            pii_entities = []
                            for item in response["Entities"]:
                                pii_data = item
                                pii_data["Text"] = extracted_text_pii[
                                    int(item["BeginOffset"]) : int(item["EndOffset"])
                                ]
                                pii_entities.append(pii_data)

                            response["Entities"] = pii_entities
                            result["piidata"] = response

                            response = COMPREHENDMEDICAL_CLIENT.detect_entities_v2(
                                Text=extracted_text_phi
                            )

                            print("AWS Comprehend PHI Response : ", response)
                            result["phidata"] = response

                        self.classification_type_details.append(result)
#                     except Exception as error:
#                         print(error)
#                         result["Classes"] = [
#                             {
# # ASP changes done for sunlife                                
#                                 # "Name": "OTHER",
#                                 "Name": "CLASSIFICATION_FAILED",
#                                 "Score": 1.0,
#                             }
#                         ]
#                        result["piidata"] = {}
#                        result["phidata"] = {}

#                        self.classification_type_details.append(result)
                else:
                    print(
                        f"Document Text Length: {len(extracted_text)}. Text in document is less than {attachmenttype_minimum_text} characters. Request Type tagged as NONDOCUMENT."
                    )
                    # result["Classes"] = [
                    #     {
                    #         "Name": "NONDOCUMENT",
                    #         "Score": 1.0,
                    #     }
                    # ]
                    result["piidata"] = {}
                    result["phidata"] = {}

                    self.classification_type_details.append(result)

                if is_document_translated == "Y":
                    copy_source = {"Bucket": bucket, "Key": translate_filename}
                    dest_filename = f"{Path(translate_filename).stem}_{submission_id}{Path(translate_filename).suffix}"
                    dest_filename = f"{Path(translate_filename).parent}/{dest_filename}"
                    archive_file(
                        copy_source=copy_source,
                        dest_bucket=bucket,
                        dest_filename=dest_filename,
                    )
                else:
                    Path.unlink(file_name)
                    self.classification_type_details[-1]["filehash"] = hash_digest
                    table_log(
                        table,
                        "update",
                        f"{submission_id}-{doc_id}",
                        "result",
# ASP fixing field size issue for inferencelog                        
#                        value=json.dumps(self.classification_type_details[-1]),
                        value="", 
                    )

                table_log(
                    table, "update", f"{submission_id}-{doc_id}", "completiontime"
                )

                if trimmed_output != "Y":
                    bucket, key = self.split_s3_path(old_attach)
                    new_key = Path(key).parent
                    upload_to_s3_asdata(
                        self.classification_type_details[-1],
                        bucket,
                        f"{new_key}/{submission_id}_{doc_id}{comprehend_json_suffix}",
                    )
                    # Upload identifier file such that we know which document was given a document id
                    upload_to_s3_asdata(
                        f"{key}_{doc_id}",
                        bucket,
                        f"{key}_{doc_id}",
                    )

            if trimmed_output != "Y":
                print("Creating consolidated output file")
                new_key = Path(key).parent
                new_suffix = comprehend_json_suffix.replace(
                    ".json", "consolidated.json"
                )
                upload_to_s3_asdata(
                    {
                        "submission_id": submission_id,
                        "result": self.classification_type_details,
                    },
                    bucket,
                    f"{new_key}/{submission_id}{new_suffix}",
                )

        return self.classification_type_details

    # def higattachment_type_classifier(
    #     self,
    #     data,
    #     submission_id,
    #     table,
    #     trimmed_output,
    #     is_document_translated,
    #     language,
    # ):
    #     print(f"Starting {self.request_type} inference")
    #     self.classification_type_details = []

    #     attachmenttype_minimum_text = int(os.getenv("ATTACHMENTTYPE_MINIMUM_TEXT"))
    #     comprehend_json_suffix = os.getenv("COMPREHEND_JSON_SUFFIX")

    #     if data:
    #         table_log(
    #             table,
    #             "update",
    #             submission_id,
    #             f"documents",
    #             value=data,
    #         )
    #         table_log(
    #             table,
    #             "update",
    #             submission_id,
    #             f"countofdocuments",
    #             value=len(data),
    #         )

    #         for filepath in data:
    #             # attach = urllib.parse.unquote_plus(attach, encoding="utf-8")
    #             if not isinstance(filepath, dict):
    #                 filepath = json.loads(filepath)

    #             attach = filepath["path"]
    #             doc_id = filepath["document_id"]
    #             table_log(table, "insert", f"{submission_id}-{doc_id}", "starttime")
    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{doc_id}",
    #                 "documents",
    #                 value=attach,
    #             )
    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{doc_id}",
    #                 "model",
    #                 value=self.request_type,
    #             )
    #             old_attach = attach
    #             if is_document_translated == "Y":
    #                 attach = f"{attach}.translate"
    #                 bucket, key = self.split_s3_path(attach)
    #                 translate_filename = key
    #                 print("Processing translated document")
    #             else:
    #                 print("Processing non-translated document")

    #             try:
    #                 bucket, key = self.split_s3_path(attach)
    #             except Exception as error:
    #                 print(f"Invalid attachment name {attach}: Error: {error}")
    #                 continue

    #             try:
    #                 data = S3_CLIENT.get_object(Bucket=bucket, Key=key)
    #                 contents = data["Body"].read()  # .decode("utf-8", "ignore")
    #             except Exception as error:
    #                 print(
    #                     "Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.".format(
    #                         key, bucket
    #                     )
    #                 )
    #                 print(error)
    #                 continue

    #             print(f"Downloaded {attach}")

    #             if is_document_translated == "Y":
    #                 extracted_text = contents.decode("utf-8", "ignore")
    #             else:
    #                 file_name = Path("/tmp").joinpath(Path(key).name)
    #                 with open(file_name, "wb") as download_file:
    #                     download_file.write(contents)
    #                     hash_digest = hashlib.sha256(contents).hexdigest()

    #                 extracted_text = ""

    #                 try:
    #                     extracted_text = self.extract_text_from_file(
    #                         file_name, hash_digest
    #                     )
    #                     table_log(
    #                         table,
    #                         "update",
    #                         f"{submission_id}-{doc_id}",
    #                         "filehash",
    #                         value=hash_digest,
    #                     )
    #                 except Exception as error:
    #                     print(error)

    #             extracted_text = extracted_text.encode("ascii", "ignore")
    #             extracted_text = extracted_text.decode()

    #             table_log(
    #                 table,
    #                 "update",
    #                 f"{submission_id}-{doc_id}",
    #                 "textract",
    #                 value=extracted_text,
    #             )

    #             extracted_text_pii = extracted_text[:99990]
    #             extracted_text_phi = extracted_text[:19990]
    #             extracted_text = extracted_text[:4990]

    #             result = {
    #                 "object_link": old_attach,
    #                 "language": language,
    #                 "document_id": doc_id,
    #             }
    #             if len(extracted_text) >= attachmenttype_minimum_text:
    #                 print(f"AWS Comprehend Request : {extracted_text}")

    #                 try:
    #                     for i in range(3):
    #                         try:
    #                             response = COMPREHEND_CLIENT.classify_document(
    #                                 Text=extracted_text, EndpointArn=self.endpoint
    #                             )
    #                             print("AWS Comprehend Response : ", response["Classes"])
    #                         except Exception as error:
    #                             print(error)
    #                             print(f"Trying #{i+1}")
    #                             time.sleep(60)
    #                             continue
    #                         break
    #                     else:
    #                         raise Exception("Maximum retries exhausted")

    #                     result["Classes"] = response["Classes"]
    #                     if trimmed_output == "Y":
    #                         print(
    #                             "PHI and PII data will be trimmed from output and no output json will be created"
    #                         )
    #                         result["piidata"] = {}
    #                         result["phidata"] = {}
    #                     else:
    #                         response = COMPREHEND_CLIENT.detect_pii_entities(
    #                             Text=extracted_text_pii, LanguageCode="en"
    #                         )

    #                         print("AWS Comprehend PII Response : ", response)

    #                         pii_entities = []
    #                         for item in response["Entities"]:
    #                             pii_data = item
    #                             pii_data["Text"] = extracted_text_pii[
    #                                 int(item["BeginOffset"]) : int(item["EndOffset"])
    #                             ]
    #                             pii_entities.append(pii_data)

    #                         response["Entities"] = pii_entities
    #                         result["piidata"] = response

    #                         response = COMPREHENDMEDICAL_CLIENT.detect_entities_v2(
    #                             Text=extracted_text_phi
    #                         )

    #                         print("AWS Comprehend PHI Response : ", response)
    #                         result["phidata"] = response

    #                     self.classification_type_details.append(result)
    #                 except Exception as error:
    #                     print(error)
    #                     result["Classes"] = [
    #                         {
    #                             "Name": "OTHER",
    #                             "Score": 1.0,
    #                         }
    #                     ]
    #                     result["piidata"] = {}
    #                     result["phidata"] = {}

    #                     self.classification_type_details.append(result)
    #             else:
    #                 print(
    #                     f"Document Text Length: {len(extracted_text)}. Text in document is less than {attachmenttype_minimum_text} characters. Request Type tagged as NONDOCUMENT."
    #                 )
    #                 result["Classes"] = [
    #                     {
    #                         "Name": "NONDOCUMENT",
    #                         "Score": 1.0,
    #                     }
    #                 ]
    #                 result["piidata"] = {}
    #                 result["phidata"] = {}

    #                 self.classification_type_details.append(result)

    #             if is_document_translated == "Y":
    #                 copy_source = {"Bucket": bucket, "Key": translate_filename}
    #                 dest_filename = f"{Path(translate_filename).stem}_{submission_id}{Path(translate_filename).suffix}"
    #                 dest_filename = f"{Path(translate_filename).parent}/{dest_filename}"
    #                 archive_file(
    #                     copy_source=copy_source,
    #                     dest_bucket=bucket,
    #                     dest_filename=dest_filename,
    #                 )
    #             else:
    #                 Path.unlink(file_name)
    #                 self.classification_type_details[-1]["filehash"] = hash_digest
    #                 table_log(
    #                     table,
    #                     "update",
    #                     f"{submission_id}-{doc_id}",
    #                     "result",
    #                     value=json.dumps(self.classification_type_details[-1]),
    #                 )

    #             table_log(
    #                 table, "update", f"{submission_id}-{doc_id}", "completiontime"
    #             )

    #             # if trimmed_output != "Y":
    #             bucket, key = self.split_s3_path(old_attach)
    #             new_key = Path(key).parent
    #             upload_to_s3_asdata(
    #                 self.classification_type_details[-1],
    #                 bucket,
    #                 f"{new_key}/{submission_id}_{doc_id}{comprehend_json_suffix}.temp",
    #             )

    #         print("Creating consolidated output file")
    #         new_key = Path(key).parent
    #         new_suffix = comprehend_json_suffix.replace(".json", "consolidated.json")
    #         upload_to_s3_asdata(
    #             {
    #                 "submission_id": submission_id,
    #                 "result": self.classification_type_details,
    #             },
    #             bucket,
    #             f"{new_key}/{submission_id}{new_suffix}",
    #         )

    #     return self.classification_type_details