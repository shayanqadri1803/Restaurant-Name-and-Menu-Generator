from dotenv import load_dotenv
from langchain_groq  import ChatGroq
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
load_dotenv()

llm = ChatGroq(
            temperature=0.5,
            groq_api_key = os.getenv("GROQ_API_KEY"),
            model_name='llama-3.1-70b-versatile'
        )




def generate_restaurant_name_and_items(cuisines):
    prompt_template_name = PromptTemplate(
    input_variables=['cuisines'],
    template='I am opening a {cuisines} restaurant give me a single fancy name for it, Do not add a preamble.'
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest some menu items for {restaurant_name}. Return them in a comma seperated list make sure there is no preamble'
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='menu_items')

    chain = SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=['cuisines'],
    output_variables=['restaurant_name', 'menu_items']
    )
    response = chain({'cuisines': cuisines})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items('Italian'))