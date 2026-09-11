"""Exact upstream evaluation protocol; whitespace is part of the model input."""

INSTRUCTION = (
    '\n'
    'You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n'
    '\n'
    '**Visual Quality:**  \n'
    'Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n'
    '- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n'
    '- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n'
    '- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n'
    '- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n'
    '- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n'
    'Textual prompt - {text_prompt}\n'
    '\n'
    '\n'
)

prompt_with_special_token = (
    '\n'
    'Please provide the overall ratings of this image: <|Reward|>\n'
    '\n'
    'END\n'
)
