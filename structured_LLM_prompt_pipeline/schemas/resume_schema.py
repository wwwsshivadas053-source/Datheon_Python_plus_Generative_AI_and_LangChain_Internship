"""Pydantic schemas for structured resume, CV, and candidate profile extraction.

Defines standardized data models for contact info, work history, skill taxonomy,
education credentials, and project portfolios.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SeniorityLevel(str, Enum):
    """Estimated candidate career seniority."""
    INTERN = "Intern"
    JUNIOR = "Junior"
    MID_LEVEL = "Mid-Level"
    SENIOR = "Senior"
    LEAD = "Lead"
    PRINCIPAL = "Principal/Staff"
    EXECUTIVE = "Executive / Director"


class PersonalInfo(BaseModel):
    """Candidate personal identity and contact points."""
    full_name: str = Field(..., description="Candidate full legal or professional name")
    email: Optional[str] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Phone number with country code if available")
    location: Optional[str] = Field(None, description="Current city, state/province, and country")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL or handle")
    github_url: Optional[str] = Field(None, description="GitHub repository profile URL")
    portfolio_url: Optional[str] = Field(None, description="Personal website or portfolio link")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: Optional[str]) -> Optional[str]:
        if v:
            v = v.strip().lower()
            if "@" not in v:
                return None
        return v


class WorkExperience(BaseModel):
    """Single employment record or role."""
    company: str = Field(..., description="Employer or organization name")
    position: str = Field(..., description="Job title or role held")
    start_date: str = Field(..., description="Start date (e.g., '2021-06' or 'June 2021')")
    end_date: Optional[str] = Field("Present", description="End date or 'Present' if current")
    is_current: bool = Field(False, description="True if this is the candidate's current employment")
    location: Optional[str] = Field(None, description="Job location or 'Remote'")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key accomplishments, quantified impacts, and responsibilities"
    )
    tech_stack_used: List[str] = Field(
        default_factory=list,
        description="Technologies, tools, and methodologies leveraged in this role"
    )


class Education(BaseModel):
    """Academic degree or qualification."""
    institution: str = Field(..., description="College, university, or school name")
    degree: str = Field(..., description="Degree type (e.g., Bachelor of Science, Master of Science, Ph.D.)")
    field_of_study: Optional[str] = Field(None, description="Major or specialization, e.g. Computer Science")
    start_year: Optional[int] = Field(None, description="Year started")
    graduation_year: Optional[int] = Field(None, description="Graduation year or expected completion")
    gpa_or_grade: Optional[str] = Field(None, description="GPA, honors, or class rank if noted")
    honors: List[str] = Field(default_factory=list, description="Academic honors, dean's list, awards")


class SkillSet(BaseModel):
    """Categorized technical and interpersonal capabilities."""
    programming_languages: List[str] = Field(default_factory=list, description="e.g., Python, TypeScript, C++, Go")
    frameworks_and_libraries: List[str] = Field(default_factory=list, description="e.g., PyTorch, React, FastAPI, LangChain")
    databases: List[str] = Field(default_factory=list, description="e.g., PostgreSQL, MongoDB, Pinecone, Redis")
    cloud_and_devops: List[str] = Field(default_factory=list, description="e.g., AWS, Docker, Kubernetes, Terraform")
    ai_and_data_tools: List[str] = Field(default_factory=list, description="e.g., Hugging Face, OpenCV, Spark, Pandas")
    soft_skills: List[str] = Field(default_factory=list, description="e.g., Team Leadership, Agile, Communication")


class Certification(BaseModel):
    """Professional licenses, badges, and credentials."""
    name: str = Field(..., description="Certification or course title")
    issuing_organization: str = Field(..., description="Issuing body, e.g., AWS, Google Cloud, DeepLearning.AI")
    issue_year: Optional[str] = Field(None, description="Year or date obtained")
    credential_id_or_url: Optional[str] = Field(None, description="Credential ID or verification link")


class Project(BaseModel):
    """Noteworthy open-source, academic, or personal projects."""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Summary of the project goals and implementation")
    role: Optional[str] = Field(None, description="Candidate's specific contribution")
    technologies: List[str] = Field(default_factory=list, description="Tools/tech utilized")
    link: Optional[str] = Field(None, description="Live link, demo, or repository URL")


class LanguageSpoken(BaseModel):
    """Spoken or natural languages."""
    language: str = Field(..., description="Language name, e.g., English, Spanish, Hindi")
    proficiency: str = Field("Proficient", description="Fluency level, e.g., Native, Fluent, Conversational")


class ResumeExtraction(BaseModel):
    """Root model for comprehensive structured resume analysis."""
    personal_info: PersonalInfo = Field(..., description="Candidate contact and social links")
    professional_summary: str = Field(..., description="Concise executive summary of candidate background")
    seniority_level: SeniorityLevel = Field(SeniorityLevel.MID_LEVEL, description="Evaluated seniority grade")
    total_years_experience: Optional[float] = Field(None, description="Calculated or stated years in industry")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Chronological work history")
    education: List[Education] = Field(default_factory=list, description="Academic qualifications")
    skills: SkillSet = Field(default_factory=SkillSet, description="Categorized skill matrix")
    certifications: List[Certification] = Field(default_factory=list, description="Professional certifications")
    projects: List[Project] = Field(default_factory=list, description="Key projects")
    languages: List[LanguageSpoken] = Field(default_factory=list, description="Natural languages")
    extraction_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0)")
