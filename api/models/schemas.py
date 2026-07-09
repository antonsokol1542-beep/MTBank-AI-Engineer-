from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TranscriptSegment(BaseModel):
    speaker: str = Field(..., description="'Оператор' or 'Клиент'")
    text: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class ClassificationResult(BaseModel):
    topic: str = Field(..., description="Main topic of the call (in Russian)")
    category: str = Field(..., description="Category: Продажи, Обслуживание, Жалоба, Консультация, Техподдержка")
    priority: Priority
    confidence: float = Field(..., ge=0.0, le=1.0)
    subcategory: Optional[str] = None


class QualityChecklist(BaseModel):
    greeting: bool = Field(..., description="Operator greeted and introduced themselves")
    need_detection: bool = Field(..., description="Operator identified client's needs")
    solution_provided: bool = Field(..., description="Operator provided a solution or escalated")
    farewell: bool = Field(..., description="Operator said a proper farewell")


class QualityScore(BaseModel):
    total: int = Field(..., ge=0, le=100, description="Quality score 0-100")
    checklist: QualityChecklist
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    passed: bool
    forbidden_phrases_found: list[str] = Field(default_factory=list)
    required_disclaimers: dict[str, bool] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    audio_duration: float = Field(..., description="Audio duration in seconds")
    transcript: list[TranscriptSegment]
    full_text: str = Field(..., description="Full transcript as plain text")
    classification: ClassificationResult
    quality_score: QualityScore
    compliance: ComplianceResult
    summary: str = Field(..., description="3-5 sentence summary of the call")
    action_items: list[str] = Field(default_factory=list)
    processing_time: float = Field(..., description="Total processing time in seconds")


class TranscribeResult(BaseModel):
    audio_duration: float
    transcript: list[TranscriptSegment]
    full_text: str
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    asr_loaded: bool
    version: str = "1.0.0"
