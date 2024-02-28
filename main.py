import gradio as gr
from vllm import LLM, SamplingParams
import torch
llm = LLM(model="TheBloke/Qwen-7B-Chat-AWQ",dtype="half",trust_remote_code=True,)
def generate_text(prompt,n):
    system_message="answer the queries"
    prompt_template=f'''{system_message}\n

    assistant
    '''
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95,top_k=40, max_tokens=n)
    outputs = llm.generate(prompt_template,sampling_params)
    for output in outputs:
        prompt=output.prompt
        generated_text=output.outputs[0].text
        return generated_text
    return outputs[0].outputs[0].text
iface = gr.Interface(
    fn=generate_text,
    inputs=["text", gr.Slider(1,200,1, label="Max Tokens")],
    outputs="text",
    title="QWEN MODEL Text Generation",
    description="Enter a prompt to generate text.",
)
iface.launch()
