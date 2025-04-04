import argparse
import importlib.util
import json
import os

from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat




def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()

def save_response(model_name, question, media_files, response, output_file="responses.csv"):
    # Convert list of media files to JSON string
    media_str = json.dumps(media_files)  # Store media as JSON string

    # Define column names
    columns = ["Model", "Media Files","Question", "Response"]

    # Check if file exists
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=columns)

    # Append the new data
    new_entry = pd.DataFrame([[model_name, question, media_str, response]], columns=columns)
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save back to CSV
    df.to_csv(output_file, index=False)

    print(f"Response saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=False, default="Efficient-Large-Model/VILA1.5-3b") 
    #Efficient-Large-Model/VILA1.5-3b
    #Llama-3-VILA1.5-8B
    parser.add_argument("--conv-mode", "-c", type=str, default="llama_3") #vicuna_v1
    """
    CONVERSATION_MODE_MAPPING = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-7b": "llama_3",
    "vila1.5-8b": "llama_3",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-40b": "hermes-2",
    "llama-3": "llama_3",
    "llama3": "llama_3",}
    """
    #Both images are taken from different angles in the same location. 
    parser.add_argument("--text", type=str, \
                        default="How many cars are in this place? Describe the color and type of each car." #from the left

                        #"The first image is at time zero, and the second is time one. Describe the colors and relative positions of the circles and squares. "
                        # 
                        #"Each photo is taken from the Different parking lot. How many cars in pictures? Describe each car's color and suv or sedan"
                        #"Each photo is taken from the same parking lot. How many cars are in the photo? Starting from the left, describe the color and type of each car, such as SUV or sedan."
                        #"Both images are taken from the same parking lot. How many cars are here? Describe the color of each car and what kind of car it is, such as an SUV or sedan."
    )
                        #"Both images are taken from different angles in the same location. How many cars this place? Describe each car's color and suv or sedan.") 
    
    parser.add_argument("--media", type=str, nargs="+", default=[
        "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4.jpg"
        # "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4_leftcrop.jpg",
        # "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4_rightcrop.jpg",
        # "image1.png",
        # "image2.png"
        
    ]) # 한 개 이상 값 들어올 수 있게 함. file1.txt file2.txt file3.txt 이런 식으로...

    """
    "/home/heejunyoon/0_DATA/go2/01197_front.png", 
    "/home/heejunyoon/0_DATA/go2/01197_right.png", 
    "/home/heejunyoon/0_DATA/go2/01197_back.png", 
    "/home/heejunyoon/0_DATA/go2/01197_left.png"

    "Which side of the man in the yellow shirt is the man on the bicycle? left? or right?"
    "Describe the scene in detail, focusing on the people in the scene."


    "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/1.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/2.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/3.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/4.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/5.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario1/6.png",
    #Describe the scene detailedly.
     #"Where is the picture hanging on the wall relative to the photo labeled 'chaplin'? left? or right?"
    #Describe the relative position of the poster 'Chaplin' and the frame on the wall"
    #"Tell me whether the poster is located to the left or right of the frame"
    #Tell me whether the chaplin poster is located to the left or right of the frame
    These are the photos of my room. Describe the relative position of the poster and the frame, and what is written in the poster.
    Describe the relative position of the poster 'Chaplin' and the power outlet
    
    
    
    
    "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75264.938.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75268.436.png", #이건 너무 어려우면 빼기
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75278.932.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75286.928.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75296.425.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75296.924.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75303.936_edit.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario2/42445991_75305.438.png",
    # These photos show the same room from different angles. how many framed pictures on the wall? describe each framed pictures.
    # 답 5개. gpt도 틀림.

    "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800773.579.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800781.076.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800773.579.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800823.576.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800825.575.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario3/42899734_800829.073.png",
    # How many framed artworks on the wall in the room? Describe each framed artwork.
    # 답 4개


    "/home/heejunyoon/0_DATA/benchmark/MSG/scenario4/41159540_39879.832.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario4/41159540_39883.831.png",
        "/home/heejunyoon/0_DATA/benchmark/MSG/scenario4/41159540_39932.327.png",

    # These pictures show the same room from different angles. how many unique wooden chairs in the room?
    #These pictures show the same room from different angles. how many unique wooden chairs in the room? Describe each chair's position
    # 답 6개 




    # "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_25_6_leftcrop.jpg",
        "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_25_6_rightcrop.jpg",
    # Both images are taken from different angles in the same location. How many unique bicycles in these pictures?

    
        "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4_leftcrop.jpg",
        "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4_rightcrop.jpg",
        # "/home/heejunyoon/0_DATA/benchmark/muirbench/isvqa_outdoor_53_4.jpg",
    #Each photo is taken from the same parking lots. How many cars in pictures? Describe each car's color and suv or sedan
    """
    parser.add_argument("--json-mode", action="store_true") # json-mode 불리면 true. output을 json으로 저장하겠다."store_true"
    parser.add_argument("--json-schema", type=str, default=None) # output json 저장 시formatting

    # parser.add_argument("--is_merging", type=bool, default=True, help="whether to merge tokens of VisionTower output")
    # # merging 위해 추가
    # parser.add_argument("--merging_mode", type=str, default="True", help="whether to merge tokens of vila output")
    args = parser.parse_args()

    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    model = llava.load(args.model_path) #model type <class 'llava.model.language_model.llava_llama.LlavaLlamaModel'>
    # import torchsummary
    # import torch
    # model = model().cuda()
    # torchsummary.summary(model, (3, 384, 384))  

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Prepare multi-modal prompt
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
            elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                media = Video(media)
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    # Generate response
    response = model.generate_content(prompt, response_format=response_format)
    print(colored(response, "cyan", attrs=["bold"]))


    import os
import pandas as pd
import json




# save_response("VILA1.5-3B", "What is AI?", media_files, "AI stands for Artificial Intelligence.")




if __name__ == "__main__":
    main()


