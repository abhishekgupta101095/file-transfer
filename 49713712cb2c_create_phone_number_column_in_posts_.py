# A template file used for generating the initial structure of new migration scripts.

"""create phone number column in posts table

Revision ID: 49713712cb2c
Revises: 
Create Date: 2024-03-12 17:42:59.186940

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '49713712cb2c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('posts', sa.Column('phone_number', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('posts', 'phone_number')

