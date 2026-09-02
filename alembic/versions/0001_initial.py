"""Initial migration - create analyses tables

Revision ID: 0001_initial
Revises: 
Create Date: 2026-01-21

"""
from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> Union[None, str]:
    op.create_table(
        'analyses',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=False, index=True),
        sa.Column('file_type', sa.String(20), nullable=False),
        sa.Column('file_size', sa.Integer, nullable=False),
        sa.Column('content_type', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'video_results',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('frames_analyzed', sa.Integer, nullable=False, default=0),
        sa.Column('faces_detected', sa.Integer, nullable=False, default=0),
        sa.Column('overall_score', sa.Float, nullable=False, default=0.5),
        sa.Column('confidence', sa.Float, nullable=False, default=0.5),
        sa.Column('is_manipulated', sa.Boolean, nullable=False, default=False),
        sa.Column('branch_scores', sa.JSON, nullable=True),
        sa.Column('per_frame_scores', sa.JSON, nullable=True),
        sa.Column('heatmap_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'audio_results',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('duration_seconds', sa.Float, nullable=False, default=0.0),
        sa.Column('segments_analyzed', sa.Integer, nullable=False, default=0),
        sa.Column('overall_score', sa.Float, nullable=False, default=0.5),
        sa.Column('confidence', sa.Float, nullable=False, default=0.5),
        sa.Column('is_manipulated', sa.Boolean, nullable=False, default=False),
        sa.Column('vocoder_detected', sa.String(50), nullable=True),
        sa.Column('artifact_evidence', sa.JSON, nullable=True),
        sa.Column('per_segment_scores', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'text_results',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('word_count', sa.Integer, nullable=False, default=0),
        sa.Column('language_detected', sa.String(10), nullable=False, default='unknown'),
        sa.Column('overall_score', sa.Float, nullable=False, default=0.5),
        sa.Column('confidence', sa.Float, nullable=False, default=0.5),
        sa.Column('is_manipulated', sa.Boolean, nullable=False, default=False),
        sa.Column('perplexity_scores', sa.JSON, nullable=True),
        sa.Column('burstiness_metrics', sa.JSON, nullable=True),
        sa.Column('per_sentence_scores', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )

    op.create_table(
        'trust_scores',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('confidence_level', sa.String(20), nullable=False),
        sa.Column('verdict', sa.String(30), nullable=False),
        sa.Column('breakdown', sa.JSON, nullable=True),
        sa.Column('fused_score', sa.Float, nullable=False),
        sa.Column('weights_used', sa.JSON, nullable=True),
        sa.Column('consistency_score', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )


def downgrade() -> Union[None, str]:
    op.drop_table('trust_scores')
    op.drop_table('text_results')
    op.drop_table('audio_results')
    op.drop_table('video_results')
    op.drop_table('analyses')
