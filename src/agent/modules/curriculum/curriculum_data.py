"""
Definición del currículo estructurado por niveles CEFR
"""

from typing import Dict, List

from .models import CEFRLevel, Competency, CompetencyType, LearningModule, LevelTransitionCriteria, SkillType


class CurriculumData:
    """Gestiona los datos del currículo estructurado"""

    @staticmethod
    def get_a1_competencies() -> List[Competency]:
        """Competencias para nivel A1 - Beginner"""
        return [
            # Basic Communication
            Competency(
                id="a1_introductions",
                name="Introductions and Personal Information",
                description="Introduce yourself and ask basic personal questions",
                level=CEFRLevel.A1,
                skill_type=SkillType.SPEAKING,
                competency_type=CompetencyType.INTRODUCTIONS,
                learning_objectives=[
                    "Say your name, age, and where you're from",
                    "Ask others about their name and origin",
                    "Use basic greetings and farewells",
                ],
                key_vocabulary=[
                    "name",
                    "age",
                    "from",
                    "live",
                    "hello",
                    "goodbye",
                    "please",
                    "thank you",
                    "yes",
                    "no",
                    "excuse me",
                    "sorry",
                ],
                grammar_points=[
                    "Present simple 'be' (I am, you are, he/she is)",
                    "Wh-questions (What, Where, How old)",
                    "Personal pronouns (I, you, he, she, it)",
                ],
                practice_activities=[
                    "Role-play introductions",
                    "Personal information exchanges",
                    "Basic conversation starters",
                ],
                assessment_criteria=[
                    "Can state name and basic information clearly",
                    "Can ask simple personal questions",
                    "Uses appropriate greetings",
                ],
                estimated_hours=3,
            ),
            # Basic Vocabulary
            Competency(
                id="a1_basic_vocabulary",
                name="Essential Daily Vocabulary",
                description="Learn fundamental words for daily life",
                level=CEFRLevel.A1,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.BASIC_VOCABULARY,
                learning_objectives=[
                    "Recognize and use 200+ basic words",
                    "Identify common objects and activities",
                    "Express basic needs and wants",
                ],
                key_vocabulary=[
                    "eat",
                    "drink",
                    "sleep",
                    "work",
                    "study",
                    "home",
                    "family",
                    "friend",
                    "food",
                    "water",
                    "money",
                    "time",
                    "day",
                    "night",
                    "good",
                    "bad",
                    "big",
                    "small",
                    "hot",
                    "cold",
                    "happy",
                    "sad",
                ],
                grammar_points=["Singular/plural nouns", "Basic adjectives", "Article usage (a, an, the)"],
                practice_activities=["Vocabulary flashcards", "Picture identification", "Basic sentence construction"],
                assessment_criteria=[
                    "Recognizes 80% of basic vocabulary",
                    "Can use words in simple sentences",
                    "Demonstrates understanding in context",
                ],
                estimated_hours=5,
            ),
            # Present Simple
            Competency(
                id="a1_present_simple",
                name="Present Simple Tense",
                description="Master basic present simple for daily routines",
                level=CEFRLevel.A1,
                skill_type=SkillType.GRAMMAR,
                competency_type=CompetencyType.PRESENT_SIMPLE,
                prerequisites=["a1_basic_vocabulary"],
                learning_objectives=[
                    "Form positive, negative, and question forms",
                    "Describe daily routines and habits",
                    "Talk about general facts",
                ],
                key_vocabulary=[
                    "wake up",
                    "get up",
                    "have breakfast",
                    "go to work",
                    "come home",
                    "watch TV",
                    "go to bed",
                    "usually",
                    "always",
                    "sometimes",
                    "never",
                ],
                grammar_points=[
                    "Present simple: I/you/we/they + verb",
                    "Present simple: he/she/it + verb + s",
                    "Do/Does for questions and negatives",
                    "Frequency adverbs",
                ],
                practice_activities=[
                    "Daily routine descriptions",
                    "Yes/No questions practice",
                    "Frequency expression exercises",
                ],
                assessment_criteria=[
                    "Correctly forms present simple sentences",
                    "Uses appropriate frequency adverbs",
                    "Can describe personal routines",
                ],
                estimated_hours=4,
            ),
            # Family and Relationships
            Competency(
                id="a1_family_vocabulary",
                name="Family and Relationships",
                description="Talk about family members and relationships",
                level=CEFRLevel.A1,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.FAMILY_VOCABULARY,
                learning_objectives=[
                    "Name family members",
                    "Describe family relationships",
                    "Talk about family activities",
                ],
                key_vocabulary=[
                    "mother",
                    "father",
                    "sister",
                    "brother",
                    "grandmother",
                    "grandfather",
                    "aunt",
                    "uncle",
                    "cousin",
                    "wife",
                    "husband",
                    "daughter",
                    "son",
                    "parents",
                    "children",
                    "married",
                    "single",
                ],
                grammar_points=[
                    "Possessive adjectives (my, your, his, her)",
                    "Have/has for relationships",
                    "Simple present for family descriptions",
                ],
                practice_activities=[
                    "Family tree creation",
                    "Family member introductions",
                    "Describing family activities",
                ],
                assessment_criteria=[
                    "Can name immediate family members",
                    "Uses possessive adjectives correctly",
                    "Can describe simple family relationships",
                ],
                estimated_hours=3,
            ),
            # Numbers and Time
            Competency(
                id="a1_numbers_time",
                name="Numbers and Time",
                description="Learn numbers, dates, and time expressions",
                level=CEFRLevel.A1,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.BASIC_VOCABULARY,
                learning_objectives=["Count from 1-100", "Tell and ask for time", "Use basic date expressions"],
                key_vocabulary=[
                    "one",
                    "two",
                    "three",
                    "ten",
                    "twenty",
                    "hundred",
                    "Monday",
                    "Tuesday",
                    "January",
                    "February",
                    "morning",
                    "afternoon",
                    "evening",
                    "night",
                    "today",
                    "tomorrow",
                    "yesterday",
                    "week",
                    "month",
                    "year",
                ],
                grammar_points=["Cardinal numbers 1-100", "Time expressions (at, in, on)", "Prepositions of time"],
                practice_activities=[
                    "Number games and exercises",
                    "Time telling practice",
                    "Date and appointment making",
                ],
                assessment_criteria=[
                    "Can count to 100 accurately",
                    "Can tell time in hours and minutes",
                    "Uses time prepositions correctly",
                ],
                estimated_hours=4,
            ),
        ]

    @staticmethod
    def get_a2_competencies() -> List[Competency]:
        """Competencias para nivel A2 - Elementary"""
        return [
            # Past Simple
            Competency(
                id="a2_past_simple",
                name="Past Simple Tense",
                description="Talk about past events and experiences",
                level=CEFRLevel.A2,
                skill_type=SkillType.GRAMMAR,
                competency_type=CompetencyType.PAST_SIMPLE,
                prerequisites=["a1_present_simple"],
                learning_objectives=[
                    "Form regular and irregular past tense",
                    "Describe past events and experiences",
                    "Ask questions about the past",
                ],
                key_vocabulary=[
                    "yesterday",
                    "last week",
                    "last year",
                    "ago",
                    "when",
                    "went",
                    "saw",
                    "came",
                    "did",
                    "was",
                    "were",
                    "had",
                ],
                grammar_points=[
                    "Regular verbs + ed",
                    "Irregular verb forms",
                    "Was/were for past states",
                    "Did for questions and negatives",
                ],
                practice_activities=["Past event narration", "Biography writing", "Past experience sharing"],
                assessment_criteria=[
                    "Uses regular and irregular past forms correctly",
                    "Can narrate simple past events",
                    "Forms past questions accurately",
                ],
                estimated_hours=5,
            ),
            # Food and Restaurants
            Competency(
                id="a2_food_vocabulary",
                name="Food and Dining",
                description="Order food and discuss eating preferences",
                level=CEFRLevel.A2,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.FOOD_VOCABULARY,
                learning_objectives=[
                    "Name common foods and drinks",
                    "Order in restaurants",
                    "Express food preferences",
                ],
                key_vocabulary=[
                    "breakfast",
                    "lunch",
                    "dinner",
                    "restaurant",
                    "menu",
                    "order",
                    "chicken",
                    "beef",
                    "fish",
                    "vegetables",
                    "rice",
                    "pasta",
                    "coffee",
                    "tea",
                    "juice",
                    "water",
                    "beer",
                    "wine",
                    "delicious",
                    "tasty",
                    "spicy",
                    "sweet",
                    "sour",
                ],
                grammar_points=[
                    "Would like for polite requests",
                    "Countable/uncountable nouns",
                    "Some/any for quantities",
                ],
                practice_activities=["Restaurant role-plays", "Menu reading exercises", "Food preference discussions"],
                assessment_criteria=[
                    "Can order food politely",
                    "Uses food vocabulary appropriately",
                    "Expresses preferences clearly",
                ],
                estimated_hours=4,
            ),
            # Travel and Directions
            Competency(
                id="a2_travel_vocabulary",
                name="Travel and Directions",
                description="Navigate and ask for directions while traveling",
                level=CEFRLevel.A2,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.TRAVEL_VOCABULARY,
                learning_objectives=[
                    "Ask for and give directions",
                    "Use travel-related vocabulary",
                    "Describe locations",
                ],
                key_vocabulary=[
                    "airport",
                    "hotel",
                    "station",
                    "bus",
                    "train",
                    "taxi",
                    "left",
                    "right",
                    "straight",
                    "turn",
                    "corner",
                    "street",
                    "near",
                    "far",
                    "next to",
                    "opposite",
                    "between",
                ],
                grammar_points=[
                    "Prepositions of place",
                    "Imperative for directions",
                    "Present continuous for current actions",
                ],
                practice_activities=[
                    "Direction giving practice",
                    "Map reading exercises",
                    "Travel planning conversations",
                ],
                assessment_criteria=[
                    "Can ask for directions clearly",
                    "Gives simple directions accurately",
                    "Uses location prepositions correctly",
                ],
                estimated_hours=4,
            ),
        ]

    @staticmethod
    def get_b1_competencies() -> List[Competency]:
        """Competencias para nivel B1 - Intermediate"""
        return [
            # Future Plans
            Competency(
                id="b1_future_simple",
                name="Future Plans and Predictions",
                description="Express future plans, predictions, and intentions",
                level=CEFRLevel.B1,
                skill_type=SkillType.GRAMMAR,
                competency_type=CompetencyType.FUTURE_SIMPLE,
                prerequisites=["a2_past_simple"],
                learning_objectives=[
                    "Use will for predictions and decisions",
                    "Use going to for planned future",
                    "Discuss future plans and goals",
                ],
                key_vocabulary=[
                    "will",
                    "going to",
                    "plan",
                    "hope",
                    "expect",
                    "predict",
                    "probably",
                    "definitely",
                    "maybe",
                    "perhaps",
                    "future",
                    "goal",
                ],
                grammar_points=[
                    "Will + infinitive for predictions",
                    "Going to + infinitive for plans",
                    "Present continuous for arranged future",
                    "Time expressions for future",
                ],
                practice_activities=["Future plans discussion", "Weather predictions", "Goal setting conversations"],
                assessment_criteria=[
                    "Distinguishes between will and going to",
                    "Expresses future plans clearly",
                    "Makes logical predictions",
                ],
                estimated_hours=5,
            ),
            # Work and Career
            Competency(
                id="b1_work_vocabulary",
                name="Work and Career",
                description="Discuss jobs, career goals, and workplace situations",
                level=CEFRLevel.B1,
                skill_type=SkillType.VOCABULARY,
                competency_type=CompetencyType.WORK_VOCABULARY,
                learning_objectives=[
                    "Describe jobs and responsibilities",
                    "Discuss career goals",
                    "Handle workplace conversations",
                ],
                key_vocabulary=[
                    "job",
                    "career",
                    "profession",
                    "salary",
                    "manager",
                    "colleague",
                    "meeting",
                    "deadline",
                    "project",
                    "responsibility",
                    "experience",
                    "qualification",
                    "interview",
                    "promotion",
                    "resign",
                    "retire",
                ],
                grammar_points=[
                    "Present perfect for experience",
                    "Modal verbs for ability and obligation",
                    "Conditional sentences for hypothetical situations",
                ],
                practice_activities=[
                    "Job interview simulations",
                    "Career goal discussions",
                    "Workplace problem solving",
                ],
                assessment_criteria=[
                    "Uses work vocabulary appropriately",
                    "Can describe job responsibilities",
                    "Handles workplace conversations confidently",
                ],
                estimated_hours=6,
            ),
        ]

    @staticmethod
    def get_learning_modules() -> Dict[CEFRLevel, List[LearningModule]]:
        """Obtiene todos los módulos de aprendizaje organizados por nivel"""
        return {
            CEFRLevel.A1: [
                LearningModule(
                    id="a1_foundations",
                    name="English Foundations",
                    description="Essential skills for complete beginners",
                    level=CEFRLevel.A1,
                    competencies=CurriculumData.get_a1_competencies(),
                    order=1,
                    is_core=True,
                )
            ],
            CEFRLevel.A2: [
                LearningModule(
                    id="a2_elementary",
                    name="Elementary Communication",
                    description="Building on basic skills for practical communication",
                    level=CEFRLevel.A2,
                    competencies=CurriculumData.get_a2_competencies(),
                    order=1,
                    is_core=True,
                )
            ],
            CEFRLevel.B1: [
                LearningModule(
                    id="b1_intermediate",
                    name="Intermediate Skills",
                    description="Developing fluency and confidence",
                    level=CEFRLevel.B1,
                    competencies=CurriculumData.get_b1_competencies(),
                    order=1,
                    is_core=True,
                )
            ],
        }

    @staticmethod
    def get_level_transition_criteria() -> List[LevelTransitionCriteria]:
        """Define los criterios para avanzar entre niveles"""
        return [
            LevelTransitionCriteria(
                from_level=CEFRLevel.A1,
                to_level=CEFRLevel.A2,
                min_competencies_completed=4,  # De 5 competencias A1
                min_average_score=0.75,
                required_core_competencies=["a1_introductions", "a1_basic_vocabulary", "a1_present_simple"],
                min_time_in_level_days=14,
            ),
            LevelTransitionCriteria(
                from_level=CEFRLevel.A2,
                to_level=CEFRLevel.B1,
                min_competencies_completed=3,  # De 3 competencias A2
                min_average_score=0.80,
                required_core_competencies=["a2_past_simple", "a2_food_vocabulary"],
                min_time_in_level_days=21,
            ),
            LevelTransitionCriteria(
                from_level=CEFRLevel.B1,
                to_level=CEFRLevel.B2,
                min_competencies_completed=2,  # De 2 competencias B1
                min_average_score=0.85,
                required_core_competencies=["b1_future_simple", "b1_work_vocabulary"],
                min_time_in_level_days=30,
            ),
        ]
