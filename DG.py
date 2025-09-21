import json
import random
import time
import os
import re
from typing import Dict, List
from dataclasses import dataclass
import google.generativeai as genai
from groq import Groq
from tqdm import tqdm
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DialogueEntry:
    question_id: str
    question: str
    ai_suggestion: str
    participant_answer: str
    bias_type: str
    confidence: int

@dataclass
class ParticipantData:
    participant_id: str
    dialogue: List[DialogueEntry]


# ------------------- API Key Manager -------------------
class APIKeyManager:
    def __init__(self, keys: List[str], service_name: str):
        self.keys = [k.strip() for k in keys if k.strip()]
        if not self.keys:
            raise ValueError(f"No API keys provided for {service_name}")
        self.index = 0
        self.service_name = service_name

    def get_key(self) -> str:
        return self.keys[self.index]

    def rotate_key(self) -> str:
        old_key = self.get_key()
        self.index = (self.index + 1) % len(self.keys)
        new_key = self.get_key()
        logger.warning(
            f"[{self.service_name}] Rotating key: {old_key[:6]}... -> {new_key[:6]}..."
        )
        return new_key


# ------------------- Main Generator -------------------
class CognitiveBiasDatasetGenerator:
    def __init__(self, gemini_keys: List[str], groq_keys: List[str], groq_model: str = "llama-3.1-8b-instant"):
        self.BIAS_TYPES = [
            "Anchoring", "Confirmation Bias", "Availability Heuristic", "Framing Effect",
            "Hindsight Bias", "Loss Aversion", "Status Quo Bias", "Optimism Bias",
            "Pessimism Bias", "Bandwagon Effect", "Sunk Cost Fallacy", "Gambler's Fallacy",
            "Overconfidence", "Halo Effect", "Self-Serving Bias", "Dunning-Kruger Effect",
            "Negativity Bias", "Survivorship Bias", "Authority Bias", "Recency Bias",
            "Selection Bias", "Outcome Bias", "False Consensus Effect", "Illusion of Control",
            "Actor-Observer Bias", "Planning Fallacy", "Just-World Hypothesis",
            "Group Attribution Error", "Pro-innovation Bias", "Spotlight Effect",
            "Illusory Correlation", "Base Rate Fallacy", "No Bias"
        ]

        # Comprehensive fallback question bank with 60 unique questions
        self.FALLBACK_QUESTIONS = [
            {"question": "Should I invest in cryptocurrency right now?", "suggestion": "AI suggests digital currencies have shown massive growth potential in recent years.", "answer": "Based on the growth patterns, I think cryptocurrency is definitely worth investing in now."},
            {"question": "Is it better to buy or rent a house?", "suggestion": "AI suggests homeownership builds long-term wealth and provides stability.", "answer": "Buying is clearly the better choice for building wealth and having control over your living space."},
            {"question": "Should I pursue a master's degree for career advancement?", "suggestion": "AI suggests advanced degrees typically lead to higher salaries and better job opportunities.", "answer": "A master's degree is definitely worth it - the investment will pay off with better career prospects."},
            {"question": "Is it safe to eat food past its expiration date?", "suggestion": "AI suggests many foods remain safe for days or weeks after their expiration dates.", "answer": "I usually eat food a few days past the date - most expiration dates are overly cautious anyway."},
            {"question": "Should I switch to a plant-based diet?", "suggestion": "AI suggests plant-based diets are linked to better health outcomes and environmental benefits.", "answer": "The health and environmental benefits make switching to plant-based eating an obvious choice."},
            {"question": "Is it worth buying extended warranties on electronics?", "suggestion": "AI suggests extended warranties provide peace of mind and protection against costly repairs.", "answer": "Extended warranties are definitely worth it - they protect you from expensive repair bills."},
            {"question": "Should I use social media for professional networking?", "suggestion": "AI suggests social media platforms have become essential tools for career development.", "answer": "Social media is absolutely crucial for networking and career growth in today's world."},
            {"question": "Is it better to pay off debt or invest money?", "suggestion": "AI suggests investing can potentially earn higher returns than the interest saved by paying off debt.", "answer": "Investing makes more sense - you can earn more in the market than you'd save on debt payments."},
            {"question": "Should I get travel insurance for international trips?", "suggestion": "AI suggests travel insurance can save thousands in case of medical emergencies or trip cancellations.", "answer": "Travel insurance is essential - the potential savings far outweigh the small premium cost."},
            {"question": "Is it worth buying organic food?", "suggestion": "AI suggests organic foods contain fewer pesticides and may have higher nutritional value.", "answer": "Organic food is definitely worth the extra cost for the health benefits and reduced chemical exposure."},
            {"question": "Should I negotiate my salary during annual reviews?", "suggestion": "AI suggests employees who negotiate earn significantly more over their careers.", "answer": "Salary negotiation is crucial - not asking is essentially leaving money on the table."},
            {"question": "Is it better to exercise in the morning or evening?", "suggestion": "AI suggests morning exercise boosts metabolism and energy levels throughout the day.", "answer": "Morning workouts are clearly superior - they give you energy and momentum for the entire day."},
            {"question": "Should I use a financial advisor for investment decisions?", "suggestion": "AI suggests professional advisors can optimize portfolios and minimize tax implications.", "answer": "A financial advisor is worth the fees - their expertise can significantly improve your returns."},
            {"question": "Is it safe to use public WiFi for banking?", "suggestion": "AI suggests public WiFi networks pose security risks for sensitive financial transactions.", "answer": "I avoid banking on public WiFi - the security risks are just too high for financial information."},
            {"question": "Should I lease or buy a car?", "suggestion": "AI suggests leasing offers lower monthly payments and access to newer technology.", "answer": "Leasing is better - you get lower payments and can always drive the latest models."},
            {"question": "Is it worth taking multivitamins daily?", "suggestion": "AI suggests multivitamins can help fill nutritional gaps in modern diets.", "answer": "Daily multivitamins are a good insurance policy against nutritional deficiencies."},
            {"question": "Should I start a side business while working full-time?", "suggestion": "AI suggests side businesses can provide additional income and entrepreneurial experience.", "answer": "A side business is a great way to diversify income and explore entrepreneurial opportunities."},
            {"question": "Is it better to save money in a bank or invest it?", "suggestion": "AI suggests inflation often outpaces savings account interest rates.", "answer": "Investing is smarter than saving - your money loses value just sitting in a bank account."},
            {"question": "Should I get a credit card to build credit history?", "suggestion": "AI suggests responsible credit card use is one of the best ways to establish good credit.", "answer": "Getting a credit card is essential for building credit history and financial opportunities."},
            {"question": "Is it worth paying extra for premium gasoline?", "suggestion": "AI suggests premium gasoline can improve engine performance and longevity.", "answer": "Premium gas is worth it for better performance and protecting your engine investment."},
            {"question": "Should I work from home or go to the office?", "suggestion": "AI suggests remote work increases productivity and improves work-life balance.", "answer": "Working from home is clearly better - increased productivity and no commute stress."},
            {"question": "Is it better to buy generic or brand-name medications?", "suggestion": "AI suggests generic medications contain the same active ingredients as brand names.", "answer": "Generic medications are just as effective and save significant money compared to brand names."},
            {"question": "Should I invest in real estate for retirement?", "suggestion": "AI suggests real estate provides passive income and appreciates over time.", "answer": "Real estate is an excellent retirement investment for steady income and long-term growth."},
            {"question": "Is it worth hiring a professional tax preparer?", "suggestion": "AI suggests tax professionals can find deductions and credits individuals often miss.", "answer": "A tax professional is worth the cost - they usually find savings that exceed their fees."},
            {"question": "Should I use dating apps to meet people?", "suggestion": "AI suggests dating apps have become the most common way couples meet.", "answer": "Dating apps are the way to go - they're how most successful relationships start nowadays."},
            {"question": "Is it better to cook at home or eat out regularly?", "suggestion": "AI suggests home cooking is healthier and more cost-effective than restaurant meals.", "answer": "Cooking at home is definitely better for both your health and your wallet."},
            {"question": "Should I get pet insurance for my dog?", "suggestion": "AI suggests pet insurance can prevent financial stress during medical emergencies.", "answer": "Pet insurance is wise - veterinary bills can be devastating without coverage."},
            {"question": "Is it worth buying a house with a homeowners association?", "suggestion": "AI suggests HOAs maintain property values and provide community amenities.", "answer": "HOAs are beneficial for maintaining property values and community standards."},
            {"question": "Should I upgrade my phone every two years?", "suggestion": "AI suggests newer phones have significantly improved features and security updates.", "answer": "Upgrading every two years keeps you current with technology and security improvements."},
            {"question": "Is it better to shop online or in physical stores?", "suggestion": "AI suggests online shopping offers better prices and convenience.", "answer": "Online shopping is superior - better prices, selection, and no dealing with crowds."},
            {"question": "Should I contribute to a 401k or Roth IRA first?", "suggestion": "AI suggests 401k contributions reduce current taxes and often include employer matching.", "answer": "The 401k is the clear priority - tax savings now plus employer matching is unbeatable."},
            {"question": "Is it worth getting a college degree in today's economy?", "suggestion": "AI suggests college graduates earn significantly more over their careers.", "answer": "A college degree is still essential - the lifetime earning potential far exceeds the cost."},
            {"question": "Should I buy term or whole life insurance?", "suggestion": "AI suggests term life insurance provides more coverage for lower premiums.", "answer": "Term life insurance is the smarter choice - better coverage at a fraction of the cost."},
            {"question": "Is it better to repair or replace old appliances?", "suggestion": "AI suggests newer appliances are more energy-efficient and reliable.", "answer": "Replacing is usually better - new appliances save money on energy and repair costs."},
            {"question": "Should I take on student loans for education?", "suggestion": "AI suggests education loans are investments in future earning potential.", "answer": "Student loans are good debt - the education will pay for itself through higher earnings."},
            {"question": "Is it worth buying solar panels for my house?", "suggestion": "AI suggests solar panels reduce electricity bills and increase home value.", "answer": "Solar panels are a great investment - they pay for themselves and add home value."},
            {"question": "Should I start investing with small amounts of money?", "suggestion": "AI suggests starting early with small amounts beats waiting to invest larger sums.", "answer": "Starting small is perfect - time in the market is more important than timing the market."},
            {"question": "Is it better to buy a new or used car?", "suggestion": "AI suggests new cars come with warranties and the latest safety features.", "answer": "New cars are worth it for the warranty, reliability, and latest safety technology."},
            {"question": "Should I get a flu shot every year?", "suggestion": "AI suggests annual flu vaccines significantly reduce illness risk and severity.", "answer": "Flu shots are definitely worth it - they're the best protection against seasonal illness."},
            {"question": "Is it worth joining a gym membership?", "suggestion": "AI suggests gym memberships provide equipment variety and motivation through community.", "answer": "Gym memberships are valuable for equipment access and the motivation of working out with others."},
            {"question": "Should I diversify my investment portfolio?", "suggestion": "AI suggests diversification reduces risk while maintaining growth potential.", "answer": "Diversification is crucial - it protects your investments while still allowing for solid growth."},
            {"question": "Is it better to book flights early or wait for deals?", "suggestion": "AI suggests booking flights 6-8 weeks in advance typically offers the best prices.", "answer": "Booking early is the way to go - waiting for deals usually results in higher prices."},
            {"question": "Should I use a password manager for online accounts?", "suggestion": "AI suggests password managers significantly improve security and convenience.", "answer": "Password managers are essential - they're the best way to stay secure online."},
            {"question": "Is it worth buying extended warranties on cars?", "suggestion": "AI suggests extended warranties can prevent unexpected major repair costs.", "answer": "Extended warranties provide peace of mind and protection against expensive repairs."},
            {"question": "Should I invest in index funds or individual stocks?", "suggestion": "AI suggests index funds provide broad diversification with lower fees.", "answer": "Index funds are the smarter choice - built-in diversification with minimal fees."},
            {"question": "Is it better to rent or buy textbooks for college?", "suggestion": "AI suggests renting textbooks can save significant money compared to buying.", "answer": "Renting is definitely better - textbooks lose value quickly and renting saves hundreds."},
            {"question": "Should I get professional photos for social media profiles?", "suggestion": "AI suggests professional photos increase engagement and credibility online.", "answer": "Professional photos are worth the investment - they make a huge difference in online presence."},
            {"question": "Is it worth buying a smart home security system?", "suggestion": "AI suggests smart security systems deter crime and provide remote monitoring capabilities.", "answer": "Smart security systems are excellent investments for safety and peace of mind."},
            {"question": "Should I take public transportation or drive to work?", "suggestion": "AI suggests public transportation reduces stress and allows productive commute time.", "answer": "Public transportation is better - less stress, environmental benefits, and time to be productive."},
            {"question": "Is it better to pay bills automatically or manually?", "suggestion": "AI suggests automatic payments prevent late fees and improve credit scores.", "answer": "Automatic payments are the way to go - they prevent missed payments and late fees."},
            {"question": "Should I buy a warranty for my laptop?", "suggestion": "AI suggests laptop warranties protect against costly repairs and replacements.", "answer": "Laptop warranties are worth it - repair costs can easily exceed the warranty price."},
            {"question": "Is it worth hiring a house cleaner?", "suggestion": "AI suggests professional cleaning saves time and often achieves better results.", "answer": "House cleaners are a great investment - they save time and do a more thorough job."},
            {"question": "Should I use a financial budgeting app?", "suggestion": "AI suggests budgeting apps help track spending and achieve financial goals.", "answer": "Budgeting apps are extremely helpful for tracking expenses and staying on financial track."},
            {"question": "Is it better to buy or lease office space for a small business?", "suggestion": "AI suggests buying office space builds equity and provides long-term stability.", "answer": "Buying is better for building equity and having control over your business space."},
            {"question": "Should I get comprehensive car insurance or just liability?", "suggestion": "AI suggests comprehensive insurance protects against theft, weather, and accident damage.", "answer": "Comprehensive insurance is worth it - it protects your investment in unexpected situations."},
            {"question": "Is it worth upgrading to first-class for long flights?", "suggestion": "AI suggests first-class provides better rest and productivity on long international flights.", "answer": "First-class is worth it for long flights - the comfort and rest make a huge difference."},
            {"question": "Should I invest in a high-yield savings account?", "suggestion": "AI suggests high-yield accounts offer better returns while maintaining liquidity.", "answer": "High-yield savings accounts are smart - better returns while keeping your money accessible."},
            {"question": "Is it better to buy generic or name-brand groceries?", "suggestion": "AI suggests generic brands often have identical quality at significantly lower prices.", "answer": "Generic brands are the smart choice - same quality for much less money."},
            {"question": "Should I get a credit monitoring service?", "suggestion": "AI suggests credit monitoring helps detect identity theft and track credit improvements.", "answer": "Credit monitoring is valuable protection against identity theft and credit problems."},
            {"question": "Is it worth buying a backup generator for my home?", "suggestion": "AI suggests backup generators provide security during power outages and extreme weather.", "answer": "Backup generators are wise investments for power security and protecting your home systems."}
        ]
        
        self.fallback_index = 0
        self.used_fallback_questions = set()

        # Key managers
        self.gemini_keys = APIKeyManager(gemini_keys, "Gemini")
        self.groq_keys = APIKeyManager(groq_keys, "Groq")

        # Init Gemini
        genai.configure(api_key=self.gemini_keys.get_key())
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 200,
                "top_p": 0.9,
            }
        )

        # Init Groq
        logger.info(f"Initializing Groq client with model: {groq_model}")
        self.groq_client = Groq(api_key=self.groq_keys.get_key())
        self.groq_model_name = groq_model

        self.participants = []
        self.used_questions = set()

    # ------------------- Enhanced Fallback System -------------------
    def get_fallback_question(self) -> Dict[str, str]:
        """Get a unique fallback question from the predefined bank."""
        available_questions = [q for q in self.FALLBACK_QUESTIONS 
                             if q["question"].lower() not in self.used_fallback_questions]
        
        if not available_questions:
            # If we've used all fallback questions, reset and start over
            logger.warning("All fallback questions used, resetting fallback bank...")
            self.used_fallback_questions.clear()
            available_questions = self.FALLBACK_QUESTIONS
        
        # Select a random question from available ones
        selected = random.choice(available_questions)
        self.used_fallback_questions.add(selected["question"].lower())
        
        return {
            "question": selected["question"],
            "ai_suggestion": selected["suggestion"],
            "fallback_answer": selected["answer"]
        }

    # ------------------- Gemini Question -------------------
    def generate_question_and_suggestion(self, max_retries: int = 5) -> Dict[str, str]:
        used_questions_str = ""
        if self.used_questions:
            sample_used = list(self.used_questions)[:10]
            used_questions_str = f"\n\nAVOID generating questions similar to these already used:\n{chr(10).join(sample_used)}\n"

        prompt = f"""You are a Cognitive Bias Question Generator Agent.

Your task is to generate realistic, everyday decision-making or judgment questions that are UNIQUE and DIFFERENT from previously generated questions.

For each sample, output in JSON with the following keys:
- "question": A clear question about a decision, choice, or belief.
- "ai_suggestion": A contextual statement or piece of information that may influence the answer (phrase it as "AI suggests...").

Constraints:
- Questions should be diverse (finance, health, education, relationships, politics, career, technology, lifestyle, food, travel, etc.).
- Each question must be UNIQUE and cover different scenarios/topics
- AI suggestions should sometimes push towards a biased conclusion, but remain subtle.
- Do not generate answers or bias type.
- Output only valid JSON format.
{used_questions_str}
Example Output:
{{
"question": "Do you think investing in tech stocks is a good idea right now?",
"ai_suggestion": "AI suggests tech companies have shown strong growth in the last 3 years."
}}

Generate one UNIQUE question-suggestion pair now:"""

        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()

                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in response")

                result = json.loads(match.group(0))
                if "question" not in result or "ai_suggestion" not in result:
                    raise KeyError("Missing required keys in response JSON")

                q_norm = result["question"].lower().strip()
                if q_norm in self.used_questions:
                    logger.warning(f"Duplicate question (attempt {attempt+1}): {result['question']}")
                    continue

                self.used_questions.add(q_norm)
                return result

            except Exception as e:
                err_msg = str(e).lower()
                if "quota" in err_msg or "429" in err_msg:
                    logger.warning("Gemini quota hit, rotating key...")
                    genai.configure(api_key=self.gemini_keys.rotate_key())
                    self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
                    continue
                logger.error(f"Gemini error (attempt {attempt+1}): {e}")

        # Enhanced fallback with unique questions
        logger.info("Using fallback question from predefined bank")
        fallback = self.get_fallback_question()
        self.used_questions.add(fallback["question"].lower())
        return {
            "question": fallback["question"], 
            "ai_suggestion": fallback["ai_suggestion"],
            "fallback_answer": fallback.get("fallback_answer")  # Pass along for potential use
        }

    # ------------------- Enhanced Groq Answer -------------------
    def generate_biased_response_with_confidence(self, question: str, ai_suggestion: str, bias_type: str, participant_id: str, fallback_answer: str = None) -> Dict[str, any]:
        bias_descriptions = {
            "Anchoring": "heavily influenced by the first piece of information encountered",
            "Confirmation Bias": "seeking information that confirms existing beliefs",
            "Availability Heuristic": "judging based on easily recalled examples",
            "Framing Effect": "influenced by how information is presented",
            "Hindsight Bias": "believing past events were more predictable than they were",
            "Loss Aversion": "preferring to avoid losses over acquiring gains",
            "Status Quo Bias": "preferring things to stay the same",
            "Optimism Bias": "overestimating positive outcomes",
            "Pessimism Bias": "overestimating negative outcomes",
            "Bandwagon Effect": "following what others are doing",
            "Sunk Cost Fallacy": "continuing because of previously invested resources",
            "Gambler's Fallacy": "believing past results affect future probabilities",
            "Overconfidence": "overestimating one's abilities or knowledge",
            "Halo Effect": "letting one positive trait influence overall judgment",
            "Self-Serving Bias": "attributing success to internal factors, failures to external",
            "Dunning-Kruger Effect": "overconfidence despite lack of competence",
            "Negativity Bias": "giving more weight to negative information",
            "Survivorship Bias": "focusing on successful examples while ignoring failures",
            "Authority Bias": "being influenced by authority figures",
            "Recency Bias": "giving more weight to recent information",
            "Selection Bias": "drawing conclusions from a non-representative sample",
            "Outcome Bias": "judging decisions based on outcomes rather than process",
            "False Consensus Effect": "overestimating how much others agree with you",
            "Illusion of Control": "overestimating one's ability to control events",
            "Actor-Observer Bias": "attributing own actions to situation, others' to personality",
            "Planning Fallacy": "underestimating time, costs, and risks of future actions",
            "Just-World Hypothesis": "believing the world is fundamentally fair",
            "Group Attribution Error": "assuming group members are similar to the group stereotype",
            "Pro-innovation Bias": "overvaluing new innovations",
            "Spotlight Effect": "overestimating how much others notice your actions",
            "Illusory Correlation": "perceiving relationships between unrelated variables",
            "Base Rate Fallacy": "ignoring general information in favor of specific information",
            "No Bias": "thinking logically and objectively without bias"
        }
        bias_desc = bias_descriptions.get(bias_type, "showing cognitive bias")

        prompt = f"""You are answering as participant {participant_id}. 

Question: {question}
Context: {ai_suggestion}

Respond as someone who is {bias_desc}. Give a natural, conversational answer that clearly demonstrates {bias_type}. 

Your answer should be 1-3 sentences and sound like a real person's response.

Answer:"""

        for attempt in range(3):
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.groq_model_name,
                    temperature=0.7,
                    max_tokens=150,
                    top_p=0.9,
                )
                answer = chat_completion.choices[0].message.content.strip()
                sentences = answer.split('. ')
                if len(sentences) > 3:
                    answer = '. '.join(sentences[:3]) + '.'
                if not answer:
                    answer = fallback_answer or "I think this decision requires careful consideration."

                confidence = self.generate_confidence_score(bias_type)
                return {"participant_answer": answer, "confidence": confidence}

            except Exception as e:
                err_msg = str(e).lower()
                if "quota" in err_msg or "429" in err_msg:
                    logger.warning("Groq quota hit, rotating key...")
                    self.groq_client = Groq(api_key=self.groq_keys.rotate_key())
                    continue
                logger.error(f"Groq error: {e}")

        # Use fallback answer if provided, otherwise use default
        final_answer = fallback_answer or "Based on the AI suggestion, I would approach this decision carefully."
        return {"participant_answer": final_answer, "confidence": random.randint(50, 80)}

    def generate_confidence_score(self, bias_type: str) -> int:
        ranges = {
            "Overconfidence": (85, 99),
            "Dunning-Kruger Effect": (90, 99),
            "Optimism Bias": (75, 90),
            "Anchoring": (70, 85),
            "Confirmation Bias": (80, 95),
            "Authority Bias": (75, 90),
            "Bandwagon Effect": (65, 85),
            "Halo Effect": (70, 85),
            "Self-Serving Bias": (80, 95),
            "Availability Heuristic": (60, 80),
            "Framing Effect": (65, 85),
            "Loss Aversion": (70, 85),
            "Status Quo Bias": (60, 75),
            "Pessimism Bias": (50, 70),
            "Negativity Bias": (55, 75),
            "Planning Fallacy": (40, 65),
            "No Bias": (60, 80)
        }
        return random.randint(*ranges.get(bias_type, (50, 80)))

    # ------------------- Dataset Generation -------------------
    def generate_dataset(self, num_participants: int = 50, questions_per_participant: int = 60) -> List[ParticipantData]:
        total_samples = num_participants * questions_per_participant
        logger.info(f"Generating {total_samples} samples with enhanced fallback system...")
        with tqdm(total=total_samples, desc="Generating dataset") as pbar:
            for p_num in range(1, num_participants + 1):
                pid = f"P{p_num:02d}"
                dialogue_entries = []
                for q_num in range(1, questions_per_participant + 1):
                    try:
                        qid = f"Q{q_num:02d}"
                        qa = self.generate_question_and_suggestion()
                        bias = random.choice(self.BIAS_TYPES)
                        resp = self.generate_biased_response_with_confidence(
                            qa["question"], 
                            qa["ai_suggestion"], 
                            bias, 
                            pid,
                            qa.get("fallback_answer")  # Pass fallback answer if available
                        )
                        dialogue_entries.append(DialogueEntry(
                            question_id=qid,
                            question=qa["question"],
                            ai_suggestion=qa["ai_suggestion"],
                            participant_answer=resp["participant_answer"],
                            bias_type=bias,
                            confidence=resp["confidence"]
                        ))
                        pbar.update(1)
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error for {pid}, Q{q_num}: {e}")
                        pbar.update(1)
                self.participants.append(ParticipantData(pid, dialogue_entries))
        
        logger.info(f"Dataset generation complete. Used {len(self.used_fallback_questions)} fallback questions.")
        return self.participants

    # ------------------- Save -------------------
    def save_dataset(self, filename: str = "cognitive_bias_dataset.json", save_individual_files: bool = True):
        dataset_json = []
        for p in self.participants:
            participant_dict = {"participant_id": p.participant_id, "dialogue": []}
            for e in p.dialogue:
                participant_dict["dialogue"].append({
                    "question_id": e.question_id,
                    "question": e.question,
                    "ai_suggestion": e.ai_suggestion,
                    "participant_answer": e.participant_answer,
                    "bias_type": e.bias_type,
                    "confidence": e.confidence
                })
            dataset_json.append(participant_dict)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dataset to {filename}")

        if save_individual_files:
            self.save_individual_participant_files()

    def save_individual_participant_files(self):
        out_dir = "Cognitive_Dataset"
        os.makedirs(out_dir, exist_ok=True)
        for p in self.participants:
            fname = os.path.join(out_dir, f"{p.participant_id}.json")
            participant_dict = {"participant_id": p.participant_id, "dialogue": []}
            for e in p.dialogue:
                participant_dict["dialogue"].append({
                    "question_id": e.question_id,
                    "question": e.question,
                    "ai_suggestion": e.ai_suggestion,
                    "participant_answer": e.participant_answer,
                    "bias_type": e.bias_type,
                    "confidence": e.confidence
                })
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(participant_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved individual files in {out_dir}/")


# ------------------- MAIN -------------------
def main():
    gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
    groq_keys = os.getenv("GROQ_API_KEYS", "").split(",")
    load_dotenv()
    if not gemini_keys or not groq_keys:
        raise ValueError("API keys not set. Please define GEMINI_API_KEYS and GROQ_API_KEYS in .env")

    NUM_PARTICIPANTS = 5
    QUESTIONS_PER_PARTICIPANT = 60

    generator = CognitiveBiasDatasetGenerator(gemini_keys, groq_keys)
    generator.generate_dataset(NUM_PARTICIPANTS, QUESTIONS_PER_PARTICIPANT)
    generator.save_dataset("cognitive_bias_dataset(46-50).json", save_individual_files=True)


if __name__ == "__main__":
    main()

# import json
# import random
# import time
# import os
# import re
# from typing import Dict, List
# from dataclasses import dataclass
# import google.generativeai as genai
# from groq import Groq
# from tqdm import tqdm
# import logging
# from dotenv import load_dotenv

# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class DialogueEntry:
#     question_id: str
#     question: str
#     ai_suggestion: str
#     participant_answer: str
#     bias_type: str
#     confidence: int

# @dataclass
# class ParticipantData:
#     participant_id: str
#     dialogue: List[DialogueEntry]


# # ------------------- API Key Manager -------------------
# class APIKeyManager:
#     def __init__(self, keys: List[str], service_name: str):
#         self.keys = [k.strip() for k in keys if k.strip()]
#         if not self.keys:
#             raise ValueError(f"No API keys provided for {service_name}")
#         self.index = 0
#         self.service_name = service_name

#     def get_key(self) -> str:
#         return self.keys[self.index]

#     def rotate_key(self) -> str:
#         old_key = self.get_key()
#         self.index = (self.index + 1) % len(self.keys)
#         new_key = self.get_key()
#         logger.warning(
#             f"[{self.service_name}] Rotating key: {old_key[:6]}... -> {new_key[:6]}..."
#         )
#         return new_key


# # ------------------- Main Generator -------------------
# class CognitiveBiasDatasetGenerator:
#     def __init__(self, gemini_keys: List[str], groq_keys: List[str], groq_model: str = "llama-3.1-8b-instant"):
#         self.BIAS_TYPES = [
#             "Anchoring", "Confirmation Bias", "Availability Heuristic", "Framing Effect",
#             "Hindsight Bias", "Loss Aversion", "Status Quo Bias", "Optimism Bias",
#             "Pessimism Bias", "Bandwagon Effect", "Sunk Cost Fallacy", "Gambler's Fallacy",
#             "Overconfidence", "Halo Effect", "Self-Serving Bias", "Dunning-Kruger Effect",
#             "Negativity Bias", "Survivorship Bias", "Authority Bias", "Recency Bias",
#             "Selection Bias", "Outcome Bias", "False Consensus Effect", "Illusion of Control",
#             "Actor-Observer Bias", "Planning Fallacy", "Just-World Hypothesis",
#             "Group Attribution Error", "Pro-innovation Bias", "Spotlight Effect",
#             "Illusory Correlation", "Base Rate Fallacy", "No Bias"
#         ]

#         # Key managers
#         self.gemini_keys = APIKeyManager(gemini_keys, "Gemini")
#         self.groq_keys = APIKeyManager(groq_keys, "Groq")

#         # Init Gemini
#         genai.configure(api_key=self.gemini_keys.get_key())
#         self.gemini_model = genai.GenerativeModel(
#             model_name="gemini-1.5-flash-latest",
#             generation_config={
#                 "temperature": 0.7,
#                 "max_output_tokens": 200,
#                 "top_p": 0.9,
#             }
#         )

#         # Init Groq
#         logger.info(f"Initializing Groq client with model: {groq_model}")
#         self.groq_client = Groq(api_key=self.groq_keys.get_key())
#         self.groq_model_name = groq_model

#         self.participants = []
#         self.used_questions = set()

#     # ------------------- Gemini Question -------------------
#     def generate_question_and_suggestion(self, max_retries: int = 5) -> Dict[str, str]:
#         used_questions_str = ""
#         if self.used_questions:
#             sample_used = list(self.used_questions)[:10]
#             used_questions_str = f"\n\nAVOID generating questions similar to these already used:\n{chr(10).join(sample_used)}\n"

#         prompt = f"""You are a Cognitive Bias Question Generator Agent.

# Your task is to generate realistic, everyday decision-making or judgment questions that are UNIQUE and DIFFERENT from previously generated questions.

# For each sample, output in JSON with the following keys:
# - "question": A clear question about a decision, choice, or belief.
# - "ai_suggestion": A contextual statement or piece of information that may influence the answer (phrase it as "AI suggests...").

# Constraints:
# - Questions should be diverse (finance, health, education, relationships, politics, career, technology, lifestyle, food, travel, etc.).
# - Each question must be UNIQUE and cover different scenarios/topics
# - AI suggestions should sometimes push towards a biased conclusion, but remain subtle.
# - Do not generate answers or bias type.
# - Output only valid JSON format.
# {used_questions_str}
# Example Output:
# {{
# "question": "Do you think investing in tech stocks is a good idea right now?",
# "ai_suggestion": "AI suggests tech companies have shown strong growth in the last 3 years."
# }}

# Generate one UNIQUE question-suggestion pair now:"""

#         for attempt in range(max_retries):
#             try:
#                 response = self.gemini_model.generate_content(prompt)
#                 response_text = response.text.strip()

#                 match = re.search(r"\{.*\}", response_text, re.DOTALL)
#                 if not match:
#                     raise ValueError("No JSON object found in response")

#                 result = json.loads(match.group(0))
#                 if "question" not in result or "ai_suggestion" not in result:
#                     raise KeyError("Missing required keys in response JSON")

#                 q_norm = result["question"].lower().strip()
#                 if q_norm in self.used_questions:
#                     logger.warning(f"Duplicate question (attempt {attempt+1}): {result['question']}")
#                     continue

#                 self.used_questions.add(q_norm)
#                 return result

#             except Exception as e:
#                 err_msg = str(e).lower()
#                 if "quota" in err_msg or "429" in err_msg:
#                     logger.warning("Gemini quota hit, rotating key...")
#                     genai.configure(api_key=self.gemini_keys.rotate_key())
#                     self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
#                     continue
#                 logger.error(f"Gemini error (attempt {attempt+1}): {e}")

#         # fallback
#         fallback_q = f"Should I make this decision about scenario {len(self.used_questions)+1}?"
#         self.used_questions.add(fallback_q.lower())
#         return {"question": fallback_q, "ai_suggestion": "AI suggests this may be a reasonable option."}

#     # ------------------- Groq Answer -------------------
#     def generate_biased_response_with_confidence(self, question: str, ai_suggestion: str, bias_type: str, participant_id: str) -> Dict[str, any]:
#         bias_descriptions = {
#             "Anchoring": "heavily influenced by the first piece of information encountered",
#             "Confirmation Bias": "seeking information that confirms existing beliefs",
#             "Availability Heuristic": "judging based on easily recalled examples",
#             "Framing Effect": "influenced by how information is presented",
#             "Hindsight Bias": "believing past events were more predictable than they were",
#             "Loss Aversion": "preferring to avoid losses over acquiring gains",
#             "Status Quo Bias": "preferring things to stay the same",
#             "Optimism Bias": "overestimating positive outcomes",
#             "Pessimism Bias": "overestimating negative outcomes",
#             "Bandwagon Effect": "following what others are doing",
#             "Sunk Cost Fallacy": "continuing because of previously invested resources",
#             "Gambler's Fallacy": "believing past results affect future probabilities",
#             "Overconfidence": "overestimating one's abilities or knowledge",
#             "Halo Effect": "letting one positive trait influence overall judgment",
#             "Self-Serving Bias": "attributing success to internal factors, failures to external",
#             "Dunning-Kruger Effect": "overconfidence despite lack of competence",
#             "Negativity Bias": "giving more weight to negative information",
#             "Survivorship Bias": "focusing on successful examples while ignoring failures",
#             "Authority Bias": "being influenced by authority figures",
#             "Recency Bias": "giving more weight to recent information",
#             "Selection Bias": "drawing conclusions from a non-representative sample",
#             "Outcome Bias": "judging decisions based on outcomes rather than process",
#             "False Consensus Effect": "overestimating how much others agree with you",
#             "Illusion of Control": "overestimating one's ability to control events",
#             "Actor-Observer Bias": "attributing own actions to situation, others' to personality",
#             "Planning Fallacy": "underestimating time, costs, and risks of future actions",
#             "Just-World Hypothesis": "believing the world is fundamentally fair",
#             "Group Attribution Error": "assuming group members are similar to the group stereotype",
#             "Pro-innovation Bias": "overvaluing new innovations",
#             "Spotlight Effect": "overestimating how much others notice your actions",
#             "Illusory Correlation": "perceiving relationships between unrelated variables",
#             "Base Rate Fallacy": "ignoring general information in favor of specific information",
#             "No Bias": "thinking logically and objectively without bias"
#         }
#         bias_desc = bias_descriptions.get(bias_type, "showing cognitive bias")

#         prompt = f"""You are answering as participant {participant_id}. 

# Question: {question}
# Context: {ai_suggestion}

# Respond as someone who is {bias_desc}. Give a natural, conversational answer that clearly demonstrates {bias_type}. 

# Your answer should be 1-3 sentences and sound like a real person's response.

# Answer:"""

#         for attempt in range(3):
#             try:
#                 chat_completion = self.groq_client.chat.completions.create(
#                     messages=[{"role": "user", "content": prompt}],
#                     model=self.groq_model_name,
#                     temperature=0.7,
#                     max_tokens=150,
#                     top_p=0.9,
#                 )
#                 answer = chat_completion.choices[0].message.content.strip()
#                 sentences = answer.split('. ')
#                 if len(sentences) > 3:
#                     answer = '. '.join(sentences[:3]) + '.'
#                 if not answer:
#                     answer = "I think this decision requires careful consideration."

#                 confidence = self.generate_confidence_score(bias_type)
#                 return {"participant_answer": answer, "confidence": confidence}

#             except Exception as e:
#                 err_msg = str(e).lower()
#                 if "quota" in err_msg or "429" in err_msg:
#                     logger.warning("Groq quota hit, rotating key...")
#                     self.groq_client = Groq(api_key=self.groq_keys.rotate_key())
#                     continue
#                 logger.error(f"Groq error: {e}")

#         return {"participant_answer": "Based on the AI suggestion, I would approach this decision carefully.", "confidence": random.randint(50, 80)}

#     def generate_confidence_score(self, bias_type: str) -> int:
#         ranges = {
#             "Overconfidence": (85, 99),
#             "Dunning-Kruger Effect": (90, 99),
#             "Optimism Bias": (75, 90),
#             "Anchoring": (70, 85),
#             "Confirmation Bias": (80, 95),
#             "Authority Bias": (75, 90),
#             "Bandwagon Effect": (65, 85),
#             "Halo Effect": (70, 85),
#             "Self-Serving Bias": (80, 95),
#             "Availability Heuristic": (60, 80),
#             "Framing Effect": (65, 85),
#             "Loss Aversion": (70, 85),
#             "Status Quo Bias": (60, 75),
#             "Pessimism Bias": (50, 70),
#             "Negativity Bias": (55, 75),
#             "Planning Fallacy": (40, 65),
#             "No Bias": (60, 80)
#         }
#         return random.randint(*ranges.get(bias_type, (50, 80)))

#     # ------------------- Dataset Generation -------------------
#     def generate_dataset(self, num_participants: int = 50, questions_per_participant: int = 60) -> List[ParticipantData]:
#         total_samples = num_participants * questions_per_participant
#         logger.info(f"Generating {total_samples} samples...")
#         with tqdm(total=total_samples, desc="Generating dataset") as pbar:
#             for p_num in range(1, num_participants + 1):
#                 pid = f"P{p_num:02d}"
#                 dialogue_entries = []
#                 for q_num in range(1, questions_per_participant + 1):
#                     try:
#                         qid = f"Q{q_num:02d}"
#                         qa = self.generate_question_and_suggestion()
#                         bias = random.choice(self.BIAS_TYPES)
#                         resp = self.generate_biased_response_with_confidence(
#                             qa["question"], qa["ai_suggestion"], bias, pid
#                         )
#                         dialogue_entries.append(DialogueEntry(
#                             question_id=qid,
#                             question=qa["question"],
#                             ai_suggestion=qa["ai_suggestion"],
#                             participant_answer=resp["participant_answer"],
#                             bias_type=bias,
#                             confidence=resp["confidence"]
#                         ))
#                         pbar.update(1)
#                         time.sleep(0.1)
#                     except Exception as e:
#                         logger.error(f"Error for {pid}, Q{q_num}: {e}")
#                         pbar.update(1)
#                 self.participants.append(ParticipantData(pid, dialogue_entries))
#         return self.participants

#     # ------------------- Save -------------------
#     def save_dataset(self, filename: str = "cognitive_bias_dataset.json", save_individual_files: bool = True):
#         dataset_json = []
#         for p in self.participants:
#             participant_dict = {"participant_id": p.participant_id, "dialogue": []}
#             for e in p.dialogue:
#                 participant_dict["dialogue"].append({
#                     "question_id": e.question_id,
#                     "question": e.question,
#                     "ai_suggestion": e.ai_suggestion,
#                     "participant_answer": e.participant_answer,
#                     "bias_type": e.bias_type,
#                     "confidence": e.confidence
#                 })
#             dataset_json.append(participant_dict)
#         with open(filename, "w", encoding="utf-8") as f:
#             json.dump(dataset_json, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved dataset to {filename}")

#         if save_individual_files:
#             self.save_individual_participant_files()

#     def save_individual_participant_files(self):
#         out_dir = "Cognitive_Dataset"
#         os.makedirs(out_dir, exist_ok=True)
#         for p in self.participants:
#             fname = os.path.join(out_dir, f"{p.participant_id}.json")
#             participant_dict = {"participant_id": p.participant_id, "dialogue": []}
#             for e in p.dialogue:
#                 participant_dict["dialogue"].append({
#                     "question_id": e.question_id,
#                     "question": e.question,
#                     "ai_suggestion": e.ai_suggestion,
#                     "participant_answer": e.participant_answer,
#                     "bias_type": e.bias_type,
#                     "confidence": e.confidence
#                 })
#             with open(fname, "w", encoding="utf-8") as f:
#                 json.dump(participant_dict, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved individual files in {out_dir}/")


# # ------------------- MAIN -------------------
# def main():
#     gemini_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
#     groq_keys = os.getenv("GROQ_API_KEYS", "").split(",")
#     load_dotenv()
#     if not gemini_keys or not groq_keys:
#         raise ValueError("API keys not set. Please define GEMINI_API_KEYS and GROQ_API_KEYS in .env")

#     NUM_PARTICIPANTS = 5
#     QUESTIONS_PER_PARTICIPANT = 60

#     generator = CognitiveBiasDatasetGenerator(gemini_keys, groq_keys)
#     generator.generate_dataset(NUM_PARTICIPANTS, QUESTIONS_PER_PARTICIPANT)
#     generator.save_dataset("cognitive_bias_dataset(11-15).json", save_individual_files=True)


# if __name__ == "__main__":
#     main()


# import json
# import random
# import time
# import os
# import re
# from typing import Dict, List
# from dataclasses import dataclass
# import google.generativeai as genai
# from groq import Groq
# from tqdm import tqdm
# import logging
# from dotenv import load_dotenv

# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class DialogueEntry:
#     question_id: str
#     question: str
#     ai_suggestion: str
#     participant_answer: str
#     bias_type: str
#     confidence: int

# @dataclass
# class ParticipantData:
#     participant_id: str
#     dialogue: List[DialogueEntry]

# class CognitiveBiasDatasetGenerator:
#     def __init__(self, gemini_api_key: str, groq_api_key: str, groq_model: str = "llama-3.1-8b-instant"):
#         """
#         Initialize the Agentic AI system for cognitive bias dataset generation.
#         """
#         self.BIAS_TYPES = [
#             "Anchoring", "Confirmation Bias", "Availability Heuristic", "Framing Effect",
#             "Hindsight Bias", "Loss Aversion", "Status Quo Bias", "Optimism Bias",
#             "Pessimism Bias", "Bandwagon Effect", "Sunk Cost Fallacy", "Gambler's Fallacy",
#             "Overconfidence", "Halo Effect", "Self-Serving Bias", "Dunning-Kruger Effect",
#             "Negativity Bias", "Survivorship Bias", "Authority Bias", "Recency Bias",
#             "Selection Bias", "Outcome Bias", "False Consensus Effect", "Illusion of Control",
#             "Actor-Observer Bias", "Planning Fallacy", "Just-World Hypothesis",
#             "Group Attribution Error", "Pro-innovation Bias", "Spotlight Effect",
#             "Illusory Correlation", "Base Rate Fallacy", "No Bias"
#         ]
        
#         # Initialize Gemini API
#         genai.configure(api_key=gemini_api_key)
#         self.gemini_model = genai.GenerativeModel(
#             model_name="gemini-1.5-flash-latest",
#             generation_config={
#             "temperature": 0.7,      # creativity
#             "max_output_tokens": 200, # keep responses short JSON
#             "top_p": 0.9,           # nucleus sampling
#         }
#         )
        
#         # Initialize Groq client
#         logger.info(f"Initializing Groq client with model: {groq_model}")
#         self.groq_client = Groq(api_key=groq_api_key)
#         self.groq_model_name = groq_model
        
#         self.participants = []
#         self.used_questions = set()  # Track used questions to avoid duplicates
        
#     def generate_question_and_suggestion(self, max_retries: int = 5) -> Dict[str, str]:
#         """
#         LLM-1: Question Generator using Gemini API with duplicate detection
#         """
#         used_questions_str = ""
#         if self.used_questions:
#             # Show a few examples of used questions to avoid duplication
#             sample_used = list(self.used_questions)[:10]
#             used_questions_str = f"\n\nAVOID generating questions similar to these already used:\n{chr(10).join(sample_used)}\n"

#         prompt = f"""You are a Cognitive Bias Question Generator Agent.

# Your task is to generate realistic, everyday decision-making or judgment questions that are UNIQUE and DIFFERENT from previously generated questions.

# For each sample, output in JSON with the following keys:
# - "question": A clear question about a decision, choice, or belief.
# - "ai_suggestion": A contextual statement or piece of information that may influence the answer (phrase it as "AI suggests...").

# Constraints:
# - Questions should be diverse (finance, health, education, relationships, politics, career, technology, lifestyle, food, travel, etc.).
# - Each question must be UNIQUE and cover different scenarios/topics
# - AI suggestions should sometimes push towards a biased conclusion, but remain subtle.
# - Do not generate answers or bias type.
# - Output only valid JSON format.
# {used_questions_str}
# Example Output:
# {{
# "question": "Do you think investing in tech stocks is a good idea right now?",
# "ai_suggestion": "AI suggests tech companies have shown strong growth in the last 3 years."
# }}

# Generate one UNIQUE question-suggestion pair now:"""

#         for attempt in range(max_retries):
#             try:
#                 response = self.gemini_model.generate_content(prompt)
#                 response_text = response.text.strip()

#                 # Extract JSON robustly
#                 match = re.search(r"\{.*\}", response_text, re.DOTALL)
#                 if not match:
#                     raise ValueError("No JSON object found in response")
                
#                 json_text = match.group(0)
#                 result = json.loads(json_text)

#                 # Validate keys
#                 if "question" not in result or "ai_suggestion" not in result:
#                     raise KeyError("Missing required keys in response JSON")

#                 # Check for duplicates
#                 question_normalized = result["question"].lower().strip()
#                 if question_normalized in self.used_questions:
#                     logger.warning(f"Duplicate question detected (attempt {attempt + 1}): {result['question']}")
#                     if attempt < max_retries - 1:
#                         continue
#                     else:
#                         # If all retries failed, modify the question slightly
#                         result["question"] = f"{result['question']} (Scenario {len(self.used_questions) + 1})"
                
#                 # Add to used questions
#                 self.used_questions.add(question_normalized)
#                 return result

#             except Exception as e:
#                 logger.error(f"Error parsing Gemini response (attempt {attempt + 1}): {e}")
#                 if attempt == max_retries - 1:
#                     # Fallback question with unique identifier
#                     fallback_question = f"Should I make this important decision about scenario {len(self.used_questions) + 1}?"
#                     self.used_questions.add(fallback_question.lower())
#                     return {
#                         "question": fallback_question,
#                         "ai_suggestion": "AI suggests several factors indicate this might be a good choice."
#                     }
    
#     def generate_biased_response_with_confidence(self, question: str, ai_suggestion: str, bias_type: str, participant_id: str) -> Dict[str, any]:
#         """
#         LLM-2: Bias Responder using Groq - now returns both answer and confidence
#         """
#         bias_descriptions = {
#             "Anchoring": "heavily influenced by the first piece of information encountered",
#             "Confirmation Bias": "seeking information that confirms existing beliefs",
#             "Availability Heuristic": "judging based on easily recalled examples",
#             "Framing Effect": "influenced by how information is presented",
#             "Hindsight Bias": "believing past events were more predictable than they were",
#             "Loss Aversion": "preferring to avoid losses over acquiring gains",
#             "Status Quo Bias": "preferring things to stay the same",
#             "Optimism Bias": "overestimating positive outcomes",
#             "Pessimism Bias": "overestimating negative outcomes",
#             "Bandwagon Effect": "following what others are doing",
#             "Sunk Cost Fallacy": "continuing because of previously invested resources",
#             "Gambler's Fallacy": "believing past results affect future probabilities",
#             "Overconfidence": "overestimating one's abilities or knowledge",
#             "Halo Effect": "letting one positive trait influence overall judgment",
#             "Self-Serving Bias": "attributing success to internal factors, failures to external",
#             "Dunning-Kruger Effect": "overconfidence despite lack of competence",
#             "Negativity Bias": "giving more weight to negative information",
#             "Survivorship Bias": "focusing on successful examples while ignoring failures",
#             "Authority Bias": "being influenced by authority figures",
#             "Recency Bias": "giving more weight to recent information",
#             "Selection Bias": "drawing conclusions from a non-representative sample",
#             "Outcome Bias": "judging decisions based on outcomes rather than process",
#             "False Consensus Effect": "overestimating how much others agree with you",
#             "Illusion of Control": "overestimating one's ability to control events",
#             "Actor-Observer Bias": "attributing own actions to situation, others' to personality",
#             "Planning Fallacy": "underestimating time, costs, and risks of future actions",
#             "Just-World Hypothesis": "believing the world is fundamentally fair",
#             "Group Attribution Error": "assuming group members are similar to the group stereotype",
#             "Pro-innovation Bias": "overvaluing new innovations",
#             "Spotlight Effect": "overestimating how much others notice your actions",
#             "Illusory Correlation": "perceiving relationships between unrelated variables",
#             "Base Rate Fallacy": "ignoring general information in favor of specific information",
#             "No Bias": "thinking logically and objectively without bias"
#         }
        
#         bias_desc = bias_descriptions.get(bias_type, "showing cognitive bias")
        
#         prompt = f"""You are answering as participant {participant_id}. 

# Question: {question}
# Context: {ai_suggestion}

# Respond as someone who is {bias_desc}. Give a natural, conversational answer that clearly demonstrates {bias_type}. 

# Your answer should be 1-3 sentences and sound like a real person's response.

# Answer:"""

#         try:
#             chat_completion = self.groq_client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 model=self.groq_model_name,
#                 temperature=0.7,
#                 max_tokens=150,
#                 top_p=0.9,
#                 stream=False,
#                 stop=None
#             )
            
#             answer = chat_completion.choices[0].message.content.strip()
            
#             # Clean up the response - take only the first few sentences
#             sentences = answer.split('. ')
#             if len(sentences) > 3:
#                 answer = '. '.join(sentences[:3]) + '.'
            
#             if not answer:
#                 answer = "I think this decision requires careful consideration based on the information provided."
            
#             # Generate confidence score based on bias type
#             confidence = self.generate_confidence_score(bias_type)
            
#             return {
#                 "participant_answer": answer,
#                 "confidence": confidence
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating biased response with Groq: {e}")
#             return {
#                 "participant_answer": f"Based on the AI suggestion, I would approach this decision carefully.",
#                 "confidence": random.randint(50, 80)
#             }
    
#     def generate_confidence_score(self, bias_type: str) -> int:
#         """
#         Generate realistic confidence scores based on bias type
#         """
#         confidence_ranges = {
#             "Overconfidence": (85, 99),
#             "Dunning-Kruger Effect": (90, 99),
#             "Optimism Bias": (75, 90),
#             "Anchoring": (70, 85),
#             "Confirmation Bias": (80, 95),
#             "Authority Bias": (75, 90),
#             "Bandwagon Effect": (65, 85),
#             "Halo Effect": (70, 85),
#             "Self-Serving Bias": (80, 95),
#             "Availability Heuristic": (60, 80),
#             "Framing Effect": (65, 85),
#             "Loss Aversion": (70, 85),
#             "Status Quo Bias": (60, 75),
#             "Pessimism Bias": (50, 70),
#             "Negativity Bias": (55, 75),
#             "Planning Fallacy": (40, 65),
#             "No Bias": (60, 80)
#         }
        
#         min_conf, max_conf = confidence_ranges.get(bias_type, (50, 80))
#         return random.randint(min_conf, max_conf)
    
#     def generate_dataset(self, num_participants: int = 50, questions_per_participant: int = 60) -> List[ParticipantData]:
#         total_samples = num_participants * questions_per_participant
#         logger.info(f"Generating dataset with {num_participants} participants, {questions_per_participant} questions each")
#         logger.info(f"Total samples to generate: {total_samples}")
        
#         with tqdm(total=total_samples, desc="Generating dataset") as pbar:
#             for participant_num in range(1, num_participants + 1):
#                 participant_id = f"P{participant_num:02d}"
#                 dialogue_entries = []
                
#                 for question_num in range(1, questions_per_participant + 1):
#                     try:
#                         question_id = f"Q{question_num:02d}"
#                         qa_data = self.generate_question_and_suggestion()
#                         bias_type = random.choice(self.BIAS_TYPES)
                        
#                         response_data = self.generate_biased_response_with_confidence(
#                             qa_data["question"], 
#                             qa_data["ai_suggestion"], 
#                             bias_type, 
#                             participant_id
#                         )
                        
#                         dialogue_entry = DialogueEntry(
#                             question_id=question_id,
#                             question=qa_data["question"],
#                             ai_suggestion=qa_data["ai_suggestion"],
#                             participant_answer=response_data["participant_answer"],
#                             bias_type=bias_type,
#                             confidence=response_data["confidence"]
#                         )
                        
#                         dialogue_entries.append(dialogue_entry)
#                         pbar.update(1)
                        
#                         # Add a small delay to respect API rate limits
#                         time.sleep(0.1)
                        
#                     except Exception as e:
#                         logger.error(f"Error generating sample for {participant_id}, question {question_num}: {e}")
#                         pbar.update(1)
#                         continue
                
#                 participant_data = ParticipantData(
#                     participant_id=participant_id,
#                     dialogue=dialogue_entries
#                 )
#                 self.participants.append(participant_data)
        
#         logger.info(f"Dataset generation completed. Generated {len(self.participants)} participants.")
#         return self.participants
    
#     def save_dataset(self, filename: str = "cognitive_bias_dataset.json", save_individual_files: bool = True):
#         # Create the main dataset file with new format
#         dataset_json = []
#         for participant in self.participants:
#             participant_dict = {
#                 "participant_id": participant.participant_id,
#                 "dialogue": []
#             }
            
#             for entry in participant.dialogue:
#                 dialogue_entry = {
#                     "question_id": entry.question_id,
#                     "question": entry.question,
#                     "ai_suggestion": entry.ai_suggestion,
#                     "participant_answer": entry.participant_answer,
#                     "bias_type": entry.bias_type,
#                     "confidence": entry.confidence
#                 }
#                 participant_dict["dialogue"].append(dialogue_entry)
            
#             dataset_json.append(participant_dict)
        
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"Combined dataset saved to {filename}")
        
#         # Create individual participant files
#         if save_individual_files:
#             self.save_individual_participant_files()
        
#         self.print_dataset_statistics()
    
#     def save_individual_participant_files(self):
#         """
#         Save individual JSON files for each participant in Cognitive_Dataset folder
#         """
#         # Create the Cognitive_Dataset folder if it doesn't exist
#         output_folder = "Cognitive_Dataset"
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#             logger.info(f"Created directory: {output_folder}")
        
#         # Save individual files for each participant
#         for participant in self.participants:
#             filename = os.path.join(output_folder, f"{participant.participant_id}.json")
            
#             participant_dict = {
#                 "participant_id": participant.participant_id,
#                 "dialogue": []
#             }
            
#             for entry in participant.dialogue:
#                 dialogue_entry = {
#                     "question_id": entry.question_id,
#                     "question": entry.question,
#                     "ai_suggestion": entry.ai_suggestion,
#                     "participant_answer": entry.participant_answer,
#                     "bias_type": entry.bias_type,
#                     "confidence": entry.confidence
#                 }
#                 participant_dict["dialogue"].append(dialogue_entry)
            
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(participant_dict, f, indent=2, ensure_ascii=False)
            
#             logger.info(f"Saved {len(participant.dialogue)} questions for {participant.participant_id} to {filename}")
        
#         logger.info(f"Individual participant files saved in {output_folder}/ directory")
#         logger.info(f"Total participants: {len(self.participants)}")
        
#         # Print summary of individual files
#         print(f"\n📁 Individual participant files created in '{output_folder}/' folder:")
#         for participant in sorted(self.participants, key=lambda x: x.participant_id):
#             sample_count = len(participant.dialogue)
#             print(f"  {participant.participant_id}.json - {sample_count} questions")
    
#     def print_dataset_statistics(self):
#         if not self.participants:
#             logger.info("No dataset to analyze")
#             return
        
#         bias_counts = {}
#         total_questions = 0
#         confidence_scores = []
        
#         for participant in self.participants:
#             for entry in participant.dialogue:
#                 total_questions += 1
#                 bias_counts[entry.bias_type] = bias_counts.get(entry.bias_type, 0) + 1
#                 confidence_scores.append(entry.confidence)
        
#         print("\n" + "="*50)
#         print("DATASET STATISTICS")
#         print("="*50)
#         print(f"Total participants: {len(self.participants)}")
#         print(f"Total questions: {total_questions}")
#         print(f"Unique bias types: {len(bias_counts)}")
        
#         if self.participants:
#             questions_per_participant = [len(p.dialogue) for p in self.participants]
#             print(f"\nParticipant distribution:")
#             print(f"Min questions per participant: {min(questions_per_participant)}")
#             print(f"Max questions per participant: {max(questions_per_participant)}")
#             print(f"Average questions per participant: {sum(questions_per_participant) / len(questions_per_participant):.1f}")
        
#         if confidence_scores:
#             print(f"\nConfidence score distribution:")
#             print(f"Min confidence: {min(confidence_scores)}")
#             print(f"Max confidence: {max(confidence_scores)}")
#             print(f"Average confidence: {sum(confidence_scores) / len(confidence_scores):.1f}")
        
#         print(f"\nBias type distribution:")
#         for bias_type, count in sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
#             print(f"  {bias_type}: {count} samples")
        
#         print(f"\nUnique questions generated: {len(self.used_questions)}")
        
#         print("\nSample entries:")
#         for i, participant in enumerate(self.participants[:2]):
#             print(f"\nParticipant {participant.participant_id}:")
#             for j, entry in enumerate(participant.dialogue[:2]):
#                 print(f"  Question {entry.question_id}:")
#                 print(f"    Q: {entry.question}")
#                 print(f"    AI: {entry.ai_suggestion}")
#                 print(f"    A: {entry.participant_answer}")
#                 print(f"    Bias: {entry.bias_type}")
#                 print(f"    Confidence: {entry.confidence}")

# def main():
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and efficient model
    
#     if not GEMINI_API_KEY:
#         raise ValueError("GEMINI_API_KEY environment variable is not set")
    
#     if not GROQ_API_KEY:
#         raise ValueError("GROQ_API_KEY environment variable is not set")
    
#     NUM_PARTICIPANTS = 1
#     QUESTIONS_PER_PARTICIPANT = 60
    
#     try:
#         generator = CognitiveBiasDatasetGenerator(
#             gemini_api_key=GEMINI_API_KEY,
#             groq_api_key=GROQ_API_KEY,
#             groq_model=GROQ_MODEL
#         )
#         dataset = generator.generate_dataset(
#             num_participants=NUM_PARTICIPANTS,
#             questions_per_participant=QUESTIONS_PER_PARTICIPANT
#         )
#         generator.save_dataset("cognitive_bias_dataset.json", save_individual_files=True)
#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
#         raise

# if __name__ == "__main__":
#     main()

