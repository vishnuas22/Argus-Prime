"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision || '000000000000'}
Create Date: ${create_date}

"""
from typing import Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> Union[None, str]:
    ${upgrades if upgrades else "pass"}


def downgrade() -> Union[None, str]:
    ${downgrades if downgrades else "pass"}
