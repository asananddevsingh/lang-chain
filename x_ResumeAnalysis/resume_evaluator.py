from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import os
from dotenv import load_dotenv
import PyPDF2
import argparse

load_dotenv()

# CLI:: python resume_evaluator.py --resume AnandDevResume.pdf --job-description senior_software_engineer_jd.txt
class ResumeScore(BaseModel):
    overall_score: float = Field(description="Overall match score between 0 and 100")
    skills_match: float = Field(description="Score for matching skills between 0 and 100")
    experience_relevance: float = Field(description="Score for relevant experience between 0 and 100")
    education_match: float = Field(description="Score for education match between 0 and 100")
    missing_requirements: List[str] = Field(description="List of missing requirements from the job description")
    matching_skills: List[str] = Field(description="List of skills that match the job description")
    analysis: List[str] = Field(description="Share the analysis of the candidate's resume against the job description")
    recommendation: str = Field(description="Small summary of the analysis and would you recommend the candidate for the role or not")

class ResumeEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=ResumeScore)
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume evaluator. Your task is to analyze a candidate's resume against a job description and provide a detailed evaluation.
            Consider the following aspects:
            1. Skills match: How well do the candidate's skills align with the required skills?
            2. Experience relevance: How relevant is the candidate's experience to the job requirements?
            3. Education match: How well does the candidate's education match the job requirements?
            4. Missing requirements: What key requirements from the job description are missing in the resume?
            5. Matching skills: What specific skills from the resume match the job requirements?
            6. Analysis: Suggest the talent acquisition team to consider the candidate for the role or not with some bullet points and reasioning
            7. Recommendation: Recommendation to the talent acquisition team to consider the candidate for the role or not
            
            Provide scores between 0 and 100 for each category.
            {format_instructions}"""),
            ("user", """Job Description:
            {job_description}
            
            Candidate's Resume: {resume}
            
            Please evaluate the resume against the job description.""")
        ])

    def evaluate_resume(self, job_description: str, resume: str) -> ResumeScore:
        """
        Evaluate a resume against a job description and return a detailed score.
        
        Args:
            job_description (str): The job description text
            resume (str): The candidate's resume text

        Returns:
            ResumeScore: A Pydantic model containing the evaluation results
        """
        prompt = self.evaluation_prompt.format_messages(
            job_description=job_description,
            resume=resume,
            format_instructions=self.parser.get_format_instructions()
        )
        
        response = self.llm(prompt)
        return self.parser.parse(response.content)

def read_pdf(file_path: str) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def read_text_file(file_path: str) -> str:
    """
    Read content from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the text file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_file(file_path: str) -> str:
    """
    Read content from a file (PDF or text).
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Content of the file
    """
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    else:
        return read_text_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a resume against a job description')
    parser.add_argument('--resume', required=True, help='Path to the resume file (PDF or text)')
    parser.add_argument('--job-description', required=True, help='Path to the job description file (PDF or text)')
    
    args = parser.parse_args()
    
    try:
        # Read the resume and job description files
        resume_text = read_file(args.resume)
        job_description_text = read_file(args.job_description)
        
        # Initialize the evaluator and get results
        evaluator = ResumeEvaluator()
        result = evaluator.evaluate_resume(job_description_text, resume_text)
        
        # Print the evaluation results
        print("\nResume Evaluation Results:")
        print(f"Overall Score: {result.overall_score}/100")
        print(f"Skills Match: {result.skills_match}/100")
        print(f"Experience Relevance: {result.experience_relevance}/100")
        print(f"Education Match: {result.education_match}/100")
        
        print("\nMissing Requirements:")
        for req in result.missing_requirements:
            print(f"- {req}")
        
        print("\nMatching Skills:")
        for skill in result.matching_skills:
            print(f"- {skill}")

        print("\nAnalysis:")
        for analysis in result.analysis:
            print(f"- {analysis}")
        
        print("\nRecommendation:") 
        print(f"- {result.recommendation}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")

if __name__ == "__main__":
    main() 