from crewai import Agent, Task, Crew
from litellm import completion

# --- Ollama Setup ---
# NOTE: You must have 'llama3' or other model pulled locally
OLLAMA_MODEL_NAME = "ollama/llama3.2-vision"

# --- Function to wrap LiteLLM call ---
def call_local_llama(prompt):
    res = completion(
        model=OLLAMA_MODEL_NAME,  # LiteLLM recognizes "ollama/llama3"
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    
    output = res["choices"][0]["message"]["content"]
    print("🧠 RAW LLM OUTPUT:\n", output)  # <--- DEBUGGING
    return output
# --- Custom LLM object that CrewAI accepts ---
class LiteLLMWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        return call_local_llama(prompt)

    def dict(self):
        return {"model": self.model, "llm_provider": "ollama"}

llm = LiteLLMWrapper(model=OLLAMA_MODEL_NAME)

# --- CrewAI Agent ---
developer_agent = Agent(
    role="Developer",
    goal="Write and execute Python code based on tasks.",
    backstory="An AI developer capable of creating Conda environments, writing code, executing it, and debugging as needed.",
    allow_code_execution=True,
    code_execution_mode="safe",
    verbose=True,
    llm=llm  # ✅ This works with CrewAI now
)

# --- Task ---
code_task = Task(
    description="Given the requirement: '{requirement}', generate Python code in a file called genereratedcode.py to fulfill it. " \
    "Create a Conda environment named crewtest, execute the code, and handle any errors by debugging and re-executing.",
    expected_output="The output of the executed Python code after successful execution.",
    agent=developer_agent
)

# --- Crew ---
crew = Crew(
    agents=[developer_agent],
    tasks=[code_task],
    verbose=True
)

# --- Run it ---
if __name__ == "__main__":
    requirement_input = "print hello world"
    result = crew.kickoff(inputs={"requirement": requirement_input})
    print("\nFinal Result:\n", result)
