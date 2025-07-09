import dspy
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Union
from datetime import datetime
import os
from dotenv import load_dotenv
import concurrent.futures
import streamlit as st

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

# Define the module (content generation orchestrator)
class CourseContentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.content_generator = ContentGenerator()

    def _generate_skill_content(self, module_name: str, skill: str):
        """Helper method for parallel execution"""
        return self.content_generator(
            module_name=module_name,
            skill_item=skill
        )

    def forward(self, modules):
        module_bundles = []
        
        for module in modules:
            # Create list of (module_name, skill) tuples for this module
            skill_tasks = [(module.module_name, skill) for skill in module.skills]
            
            # Execute all skills for this module in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self._generate_skill_content, module_name, skill)
                    for module_name, skill in skill_tasks
                ]
                
                # Collect results
                content_blocks = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    content_blocks.extend(result.content_blocks)
            
            # Create ModuleContentBundle
            bundle = ModuleContentBundle(
                module_name=module.module_name,
                content_blocks=content_blocks
            )
            module_bundles.append(bundle)
        
        # Return CourseContentResult
        return CourseContentResult(modules=module_bundles)



#STREAMLIT FRONTEND
def display_course_input():
    """Display the hardcoded course input parameters"""
    st.header("📚 AI Course Generator")
    st.markdown("Add description here")
    
    with st.expander("📝 Course Input Parameters (Hardcoded)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Course Title:**")
            st.info(sample_course_prompt.course_title)
            
            st.markdown("**Course Topics:**")
            st.info(sample_course_prompt.course_topics)
            
            st.markdown("**Course Description:**")
            st.info(sample_course_prompt.course_description)
        
        with col2:
            st.markdown("**Starting Point:**")
            st.info(sample_course_prompt.starting_point_description)
            
            st.markdown("**Finish Line:**")
            st.info(sample_course_prompt.finish_line_description)

def display_knowledge_gaps(skills_list):
    """Display the knowledge gap analysis results"""
    st.subheader("🎯 Knowledge Gap Analysis Results")
    st.markdown(f"**Total skills identified:** {len(skills_list)}")
    
    for i, skill in enumerate(skills_list, 1):
        st.markdown(f"{i}. {skill}")

def display_modules(modules):
    """Display the module grouping results"""
    st.subheader("📋 Course Module Structure")
    
    for i, module in enumerate(modules, 1):
        with st.expander(f"Module {i}: {module.module_name}", expanded=True):
            st.markdown("**Skills covered:**")
            for skill in module.skills:
                st.markdown(f"• {skill}")

def display_content_block(content_block, block_num):
    """Display a single content block"""
    with st.container():
        st.markdown(f"**Content Block {block_num}: {content_block.title}**")
        
        if content_block.type == "Text":
            st.markdown(content_block.body)
        
        elif content_block.type == "Question":
            st.markdown(f"**Question:** {content_block.question_text}")
            
            # Create interactive quiz
            quiz_key = f"quiz_{block_num}_{content_block.title}"
            user_answer = st.radio(
                "Choose your answer:",
                content_block.options,
                key=quiz_key,
                index=None
            )
            
            if user_answer:
                if user_answer == content_block.correct_answer:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {content_block.correct_answer}")
        
        st.divider()

def display_final_course(course_content_result):
    """Display the complete generated course"""
    st.header("🎓 Generated Course Content")
    
    # Course overview
    total_modules = len(course_content_result.modules)
    total_content_blocks = sum(len(module.content_blocks) for module in course_content_result.modules)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Modules", total_modules)
    with col2:
        st.metric("Total Content Blocks", total_content_blocks)
    with col3:
        text_blocks = sum(1 for module in course_content_result.modules for block in module.content_blocks if block.type == "Text")
        question_blocks = total_content_blocks - text_blocks
        st.metric("Quiz Questions", question_blocks)
    
    # Display each module
    for i, module_bundle in enumerate(course_content_result.modules, 1):
        with st.expander(f"📖 Module {i}: {module_bundle.module_name}", expanded=False):
            st.markdown(f"**Content blocks in this module:** {len(module_bundle.content_blocks)}")
            
            for j, content_block in enumerate(module_bundle.content_blocks, 1):
                display_content_block(content_block, j)

def run_course_generation():
    """Run the complete course generation process with live updates"""
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Knowledge Gap Analysis
    status_text.text("🔍 Step 1: Analyzing knowledge gaps...")
    progress_bar.progress(10)
    
    with st.status("Running Knowledge Gap Analysis...", expanded=True) as status:
        analyzer = KnowledgeGapAnalyzer()
        result = analyzer(
            starting_point_description=sample_course_prompt.starting_point_description,
            finish_line_description=sample_course_prompt.finish_line_description
        )
        skills_list = result.analysis.knowledge_skills_list
        status.update(label="✅ Knowledge Gap Analysis Complete!", state="complete")
    
    progress_bar.progress(33)
    
    # Display Step 1 results
    display_knowledge_gaps(skills_list)
    
    # Step 2: Module Grouping
    status_text.text("📋 Step 2: Grouping skills into modules...")
    
    with st.status("Running Module Grouping...", expanded=True) as status:
        grouper = ModuleGrouper()
        grouping_result = grouper(knowledge_skills_list=skills_list)
        modules = grouping_result.grouping.modules
        status.update(label="✅ Module Grouping Complete!", state="complete")
    
    progress_bar.progress(66)
    
    # Display Step 2 results
    display_modules(modules)
    
    # Step 3: Content Generation
    status_text.text("✍️ Step 3: Generating course content...")
    
    with st.status("Running Content Generation...", expanded=True) as status:
        course_generator = CourseContentGenerator()
        course_content_result = course_generator(modules=modules)
        status.update(label="✅ Content Generation Complete!", state="complete")
    
    progress_bar.progress(100)
    status_text.text("🎉 Course generation complete!")
    
    # Display final results
    display_final_course(course_content_result)
    
    # Store results in session state
    st.session_state.skills_list = skills_list
    st.session_state.modules = modules
    st.session_state.course_content = course_content_result
    st.session_state.generation_complete = True

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI Course Generator",
        page_icon="🎓",
        layout="wide"
    )
    
    # Display course input
    display_course_input()
    
    # Generation button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Course", type="primary", use_container_width=True):
            st.session_state.generation_started = True
            # Clear previous results
            if 'skills_list' in st.session_state:
                del st.session_state.skills_list
            if 'modules' in st.session_state:
                del st.session_state.modules
            if 'course_content' in st.session_state:
                del st.session_state.course_content
            if 'generation_complete' in st.session_state:
                del st.session_state.generation_complete
    
    # Run generation if button was clicked
    if st.session_state.get('generation_started', False):
        if not st.session_state.get('generation_complete', False):
            run_course_generation()
        else:
            # Display cached results
            st.success("Course already generated! Here are the results:")
            display_knowledge_gaps(st.session_state.skills_list)
            display_modules(st.session_state.modules)
            display_final_course(st.session_state.course_content)
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()