from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Elon Musk is the wealthiest person in the world, with an estimated net worth of US$676 billion as of February 2026, according to the Bloomberg Billionaires Index,[1] and $852 billion according to Forbes,[2] primarily from his ownership stakes in SpaceX and Tesla.

Having been first listed on the Forbes Billionaires List in 2012,[3] around 75% of Musk's wealth was derived from Tesla stock in November 2020,[4] although he describes himself as "cash poor".[5][6] According to Forbes, he became the first person in the world to achieve a net worth of $300 billion in 2021; $400 billion in December 2024;[7] $500 billion in October 2025;[8] $600 billion in mid-December 2025;[9] $700 billion later that month;[10] and $800 billion in February 2026.[11]

In November 2025, a Tesla pay package worth potentially $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.[12]
    """
    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two intresting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm= ChatOllama (temperature= 0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
