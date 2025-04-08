CREATE TABLE organizations (
    id UUID PRIMARY KEY, 
    name VARCHAR,
    products_enabled UUID[], 
    primary_contact VARCHAR
);

CREATE TABLE products (
    id UUID PRIMARY KEY, 
    name VARCHAR,
    create_user_api VARCHAR,
    edit_user_api VARCHAR,
    delete_user_api VARCHAR,
    is_teams_required boolean,
    create_team_api VARCHAR,
    list_team_api VARCHAR,
    is_custom_role_required boolean 
);

CREATE TABLE user_types (
    id UUID PRIMARY KEY,
    name VARCHAR
);

CREATE TABLE custom_roles (
    id UUID PRIMARY KEY,
    name VARCHAR,
    product_id UUID,
    value VARCHAR,
	FOREIGN KEY(product_id) REFERENCES products(id)
);

CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR,
    org_id UUID,
    user_type_id UUID,
	FOREIGN KEY(org_id) REFERENCES organizations(id),
	FOREIGN KEY(user_type_id) REFERENCES user_types(id)
);

CREATE TABLE personas (
    id UUID PRIMARY KEY,
    name VARCHAR,
    product_id UUID,
	FOREIGN KEY(product_id) REFERENCES products(id)
);

CREATE TABLE teams (
    id UUID PRIMARY KEY,
    name VARCHAR,
    org_id UUID,
	FOREIGN KEY(org_id) REFERENCES organizations(id)
);

-- CREATE TABLE user_mappings (
--     id UUID PRIMARY KEY,
--     user_id UUID,
--     product_id UUID,
--     team_ids UUID[],
--     persona_ids UUID[],
--     custom_role_id UUID[],
-- 	FOREIGN KEY(user_id) REFERENCES users(id),
-- 	FOREIGN KEY(product_id) REFERENCES products(id),
-- 	FOREIGN KEY(team_ids) REFERENCES teams(id),
-- 	FOREIGN KEY(persona_ids) REFERENCES personas(id),
-- 	FOREIGN KEY(custom_role_id) REFERENCES custom_roles(id)
-- );

CREATE TABLE user_team_mappings (
    user_mapping_id UUID,
    team_id UUID,
    PRIMARY KEY (user_mapping_id, team_id),
    FOREIGN KEY (user_mapping_id) REFERENCES user_mappings(id),
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE user_persona_mappings (
    user_mapping_id UUID,
    persona_id UUID,
    PRIMARY KEY (user_mapping_id, persona_id),
    FOREIGN KEY (user_mapping_id) REFERENCES user_mappings(id),
    FOREIGN KEY (persona_id) REFERENCES personas(id)
);

CREATE TABLE user_custom_role_mappings (
    user_mapping_id UUID,
    custom_role_id UUID,
    PRIMARY KEY (user_mapping_id, custom_role_id),
    FOREIGN KEY (user_mapping_id) REFERENCES user_mappings(id),
    FOREIGN KEY (custom_role_id) REFERENCES custom_roles(id)
);


CREATE TABLE user_mappings (
    id UUID PRIMARY KEY,
    user_id UUID,
    product_id UUID,
	FOREIGN KEY(user_id) REFERENCES users(id),
	FOREIGN KEY(product_id) REFERENCES products(id)
);
