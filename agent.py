import dspy
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Union
from datetime import datetime
import os
from dotenv import load_dotenv

#load env vars from .env
load_dotenv()


class CoursePrompt(BaseModel):
    """Input structure for course generation prompt"""
    course_title: str
    course_topics: str
    course_description: str
    starting_point_description: str
    finish_line_description: str


# Add the structured output model
class KnowledgeGapResult(BaseModel):
    """Structured output for knowledge gap analysis"""
    knowledge_skills_list: List[str]


# Hardcoded sample input from user
sample_course_prompt = CoursePrompt(
    course_title="Web Development for Beginners",
    course_topics="HTML structure, CSS styling, basic JavaScript, responsive design, web hosting basics",
    course_description="A comprehensive introduction to web development for complete beginners. This course covers fundamental web technologies and practical skills needed to create and deploy a simple website.",
    starting_point_description="No prior programming experience required. Students should have basic computer literacy and be comfortable using a text editor.",
    finish_line_description="Students will be able to create a simple website using HTML, CSS, and basic JavaScript, understand core web development concepts, deploy their website, and be prepared to learn more advanced web development topics."
)



lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key=os.getenv('ANTHROPIC_API_KEY'))
dspy.configure(lm=lm)



#STEP 1: analyze knowledge gap
# Define the signature
class KnowledgeGapSignature(dspy.Signature):
    """Analyze the starting point and finish line to identify the knowledge and skills needed to bridge the gap."""
    starting_point_description = dspy.InputField(desc="Description of the student's initial knowledge and skills")
    finish_line_description = dspy.InputField(desc="Description of the knowledge and skills the student should acquire")
    analysis: KnowledgeGapResult = dspy.OutputField(desc="Structured analysis of knowledge and skills needed to bridge the gap. Return a list where each item is a specific knowledge area or skill that the student needs to learn.")

# Define the module
class KnowledgeGapAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(KnowledgeGapSignature)

    def forward(self, starting_point_description, finish_line_description):
        return self.predictor(
            starting_point_description=starting_point_description,
            finish_line_description=finish_line_description
        )

#STEP 2: group knowledge gap items into modules
# Define the output structure
class Module(BaseModel):
    module_name: str
    skills: List[str]

class ModuleGroupingResult(BaseModel):
    modules: List[Module]

# Define the signature
class ModuleGroupingSignature(dspy.Signature):
    """Group the list of knowledge and skills into coherent modules for the course."""
    knowledge_skills_list = dspy.InputField(desc="List of knowledge and skills needed to bridge the gap")
    grouping: ModuleGroupingResult = dspy.OutputField(desc="Structured grouping of knowledge and skills into modules")

# Define the module
class ModuleGrouper(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ModuleGroupingSignature)

    def forward(self, knowledge_skills_list):
        return self.predictor(knowledge_skills_list=knowledge_skills_list)

#STEP 3: generate content for each item in the list in each module
# Define the output structure
# ───────────────────────────────
# Shared base (keeps DB metadata)
# ───────────────────────────────
class _BaseContentOut(BaseModel):
    # Remove ConfigDict that causes DSPy parsing issues
    # model_config = ConfigDict(from_attributes=True)

    # Make database metadata fields optional with defaults so LM doesn't generate them
    id: Optional[int] = Field(default=1)
    title: str
    is_complete: Optional[bool] = Field(default=True, alias="isComplete")
    module_id: Optional[int] = Field(default=1, alias="moduleId")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, alias="createdAt")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, alias="updatedAt")

# ───────────────────────────────
# 1. Text-only content
# ───────────────────────────────
class TextContentOut(_BaseContentOut):
    """Structured output for a 'Text' content block."""
    type: Literal["Text"] = "Text"          # keep if you still store/serve both variants
    body: str

# ───────────────────────────────
# 2. Multiple-choice question
# ───────────────────────────────
class QuestionContentOut(_BaseContentOut):
    """Structured output for a 'Question' content block."""
    type: Literal["Question"] = "Question"  # idem
    question_text: str = Field(alias="questionText")
    options: List[str]
    correct_answer: str = Field(alias="correctAnswer")
    user_answer: Optional[str] = Field(default=None, alias="userAnswer")

# Define the signature (core content generator)
class ContentGeneratorSignature(dspy.Signature):
    """Generate educational content for a specific skill within a module. Can output multiple content blocks including text explanations and quiz questions."""
    module_name = dspy.InputField(desc="Name of the module this content belongs to")
    skill_item = dspy.InputField(desc="Specific knowledge/skill item to create content for")
    content_blocks: List[Union[TextContentOut, QuestionContentOut]] = dspy.OutputField(desc="Generated content blocks - can include both explanatory text and quiz questions")

# Define the module (core content generator)
class ContentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ContentGeneratorSignature)

    def forward(self, module_name, skill_item):
        return self.predictor(
            module_name=module_name,
            skill_item=skill_item
        )

# Define the signature (content generation orchestrator)
# Re-use the two content block models you already defined:
#   TextContentOut, QuestionContentOut   (import or keep in same file)
class ModuleContentBundle(BaseModel):
    """One module plus every content block generated for its skills."""
    module_name: str
    content_blocks: List[Union[TextContentOut, QuestionContentOut]]

class CourseContentResult(BaseModel):
    """Full course payload: list of ModuleContentBundle objects."""
    modules: List[ModuleContentBundle]

class CourseContentGeneratorSignature(dspy.Signature):
    """
    Generate all content for all modules in a single shot.

    Input
    -----
    modules : List[Module]
        Each Module has `module_name` and `skills` (List[str]).

    Output
    ------
    course_content : CourseContentResult
        For every Module, returns a `ModuleContentBundle` whose
        `content_blocks` list contains one Text/Question object per skill,
        produced by your downstream ContentGenerator module.
    """
    modules = dspy.InputField(desc="List of Modules with their skills")
    course_content: CourseContentResult = dspy.OutputField(
        desc="Structured course content for every skill in every module"
    )

# Define the module (content generation orchestrator)
class CourseContentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.content_generator = ContentGenerator()

    def forward(self, modules):
        module_bundles = []
        
        for module in modules:
            content_blocks = []
            for skill in module.skills:
                # Call ContentGenerator for each skill
                result = self.content_generator(
                    module_name=module.module_name,
                    skill_item=skill
                )
                content_blocks.extend(result.content_blocks)
            
            # Create ModuleContentBundle
            bundle = ModuleContentBundle(
                module_name=module.module_name,
                content_blocks=content_blocks
            )
            module_bundles.append(bundle)
        
        # Return CourseContentResult
        return CourseContentResult(modules=module_bundles)



#TEST CODE
# Test the knowledge gap analyzer
if __name__ == "__main__":
    # Do STEP 1
    # Create analyzer instance
    analyzer = KnowledgeGapAnalyzer()
    
    # Run the analysis
    result = analyzer(
        starting_point_description=sample_course_prompt.starting_point_description,
        finish_line_description=sample_course_prompt.finish_line_description
    )
    
    # Print the results from STEP 1
    print("=== KNOWLEDGE GAP ANALYSIS ===")
    print(f"Starting Point: {sample_course_prompt.starting_point_description}")
    print(f"Finish Line: {sample_course_prompt.finish_line_description}")
    print(f"\nKnowledge/Skills Needed:")
    
    # Now you can access the structured list
    skills_list = result.analysis.knowledge_skills_list
    for i, skill in enumerate(skills_list, 1):
        print(f"{i}. {skill}")
    
    # For step 2, you can now easily iterate over the list
    print(f"\nTotal skills identified: {len(skills_list)}")
    print("Skills for processing in Step 2:")
    for skill in skills_list:
        print(f"  - {skill}")
    

    # Do STEP 2
    # Test the module grouper
    grouper = ModuleGrouper()
    grouping_result = grouper(knowledge_skills_list=skills_list)

    # Print the results from STEP 2
    # Print the grouped modules
    print("\n=== MODULE GROUPING ===")
    for i, module in enumerate(grouping_result.grouping.modules, 1):
        print(f"Module {i}: {module.module_name}")
        for skill in module.skills:
            print(f"  - {skill}")
    

    # Do STEP 3
    # Test the course content generator
    course_generator = CourseContentGenerator()
    course_content_result = course_generator(modules=grouping_result.grouping.modules)

    # Print the results from STEP 3
    print("\n=== COURSE CONTENT GENERATION ===")
    for i, module_bundle in enumerate(course_content_result.modules, 1):
        print(f"\nModule {i}: {module_bundle.module_name}")
        print(f"Content blocks generated: {len(module_bundle.content_blocks)}")
        
        for j, content_block in enumerate(module_bundle.content_blocks, 1):
            print(f"  Content Block {j}:")
            print(f"    Type: {content_block.type}")
            print(f"    Title: {content_block.title}")
            
            if content_block.type == "Text":
                print(f"    Body: {content_block.body[:100]}..." if len(content_block.body) > 100 else f"    Body: {content_block.body}")
            elif content_block.type == "Question":
                print(f"    Question: {content_block.question_text}")
                print(f"    Options: {content_block.options}")
                print(f"    Correct Answer: {content_block.correct_answer}")