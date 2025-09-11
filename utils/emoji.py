import emoji
import random

emoji_shortcodes = [
    ":grinning_face_with_big_eyes:",
    ":smiling_face_with_smiling_eyes:",
    ":face_with_tears_of_joy:",
    ":rolling_on_the_floor_laughing:",
    ":smiling_face_with_heart-eyes:",
    ":thinking_face:",
    ":face_with_raised_eyebrow:",
    ":unamused_face:",
    ":pouting_face:",
    ":angry_face:",
    ":pile_of_poo:",
    ":red_heart:",
    ":thumbs_up:",
    ":clapping_hands:",
    ":rocket:",
    ":star:",
    ":sun:",
    ":moon:",
    ":pizza:",
    ":dog_face:",
    ":cat_face:",
    ":panda_face:"
]

def generate_random_emoji():
    random_shortcode = random.choice(emoji_shortcodes)
    return emoji.emojize(random_shortcode, language='en')
