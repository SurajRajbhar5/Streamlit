import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi as yta
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

LANGS = ['ENGLISH','HINDI','MARATHI','BENGALI', 'PUNJABI', 'AFRIKAANS', 'ALBANIAN', 'AMHARIC', 'ARABIC', 'ARMENIAN', 'ASSAMESE', 'AYMARA', 'AZERBAIJANI', 'BAMBARA', 'BASQUE', 'BELARUSIAN', 'BHOJPURI', 'BOSNIAN', 'BULGARIAN', 'CATALAN', 'CEBUANO', 'CHICHEWA', 'CHINESE (SIMPLIFIED)', 'CHINESE (TRADITIONAL)', 'CORSICAN', 'CROATIAN', 'CZECH', 'DANISH', 'DHIVEHI', 'DOGRI', 'DUTCH', 'ESPERANTO', 'ESTONIAN', 'EWE', 'FILIPINO', 'FINNISH', 'FRENCH', 'FRISIAN', 'GALICIAN', 'GEORGIAN', 'GERMAN', 'GREEK', 'GUARANI', 'GUJARATI', 'HAITIAN CREOLE', 'HAUSA', 'HAWAIIAN', 'HEBREW', 'HMONG', 'HUNGARIAN', 'ICELANDIC', 'IGBO', 'ILOCANO', 'INDONESIAN', 'IRISH', 'ITALIAN', 'JAPANESE', 'JAVANESE', 'KANNADA', 'KAZAKH', 'KHMER', 'KINYARWANDA', 'KONKANI', 'KOREAN', 'KRIO', 'KURDISH (KURMANJI)', 'KURDISH (SORANI)', 'KYRGYZ', 'LAO', 'LATIN', 'LATVIAN', 'LINGALA', 'LITHUANIAN', 'LUGANDA', 'LUXEMBOURGISH', 'MACEDONIAN', 'MAITHILI', 'MALAGASY', 'MALAY', 'MALAYALAM', 'MALTESE', 'MAORI', 'MEITEILON (MANIPURI)', 'MIZO', 'MONGOLIAN', 'MYANMAR', 'NEPALI', 'NORWEGIAN', 'ODIA (ORIYA)', 'OROMO', 'PASHTO', 'PERSIAN', 'POLISH', 'PORTUGUESE', 'QUECHUA', 'ROMANIAN', 'RUSSIAN', 'SAMOAN', 'SANSKRIT', 'SCOTS GAELIC', 'SEPEDI', 'SERBIAN', 'SESOTHO', 'SHONA', 'SINDHI', 'SINHALA', 'SLOVAK', 'SLOVENIAN', 'SOMALI', 'SPANISH', 'SUNDANESE', 'SWAHILI', 'SWEDISH', 'TAJIK', 'TAMIL', 'TATAR', 'TELUGU', 'THAI', 'TIGRINYA', 'TSONGA', 'TURKISH', 'TURKMEN', 'TWI', 'UKRAINIAN', 'URDU', 'UYGHUR', 'UZBEK', 'VIETNAMESE', 'WELSH', 'XHOSA', 'YIDDISH', 'YORUBA', 'ZULU']
Langs_dict = {'AFRIKAANS': 'af', 'ALBANIAN': 'sq', 'AMHARIC': 'am', 'ARABIC': 'ar', 'ARMENIAN': 'hy', 'ASSAMESE': 'as', 'AYMARA': 'ay', 'AZERBAIJANI': 'az', 'BAMBARA': 'bm', 'BASQUE': 'eu', 'BELARUSIAN': 'be', 'BENGALI': 'bn', 'BHOJPURI': 'bho', 'BOSNIAN': 'bs', 'BULGARIAN': 'bg', 'CATALAN': 'ca', 'CEBUANO': 'ceb', 'CHICHEWA': 'ny', 'CHINESE (SIMPLIFIED)': 'zh-cn', 'CHINESE (TRADITIONAL)': 'zh-tw', 'CORSICAN': 'co', 'CROATIAN': 'hr', 'CZECH': 'cs', 'DANISH': 'da', 'DHIVEHI': 'dv', 'DOGRI': 'doi', 'DUTCH': 'nl', 'ENGLISH': 'en', 'ESPERANTO': 'eo', 'ESTONIAN': 'et', 'EWE': 'ee', 'FILIPINO': 'tl', 'FINNISH': 'fi', 'FRENCH': 'fr', 'FRISIAN': 'fy', 'GALICIAN': 'gl', 'GEORGIAN': 'ka', 'GERMAN': 'de', 'GREEK': 'el', 'GUARANI': 'gn', 'GUJARATI': 'gu', 'HAITIAN CREOLE': 'ht', 'HAUSA': 'ha', 'HAWAIIAN': 'haw', 'HEBREW': 'iw', 'HINDI': 'hi', 'HMONG': 'hmn', 'HUNGARIAN': 'hu', 'ICELANDIC': 'is', 'IGBO': 'ig', 'ILOCANO': 'ilo', 'INDONESIAN': 'id', 'IRISH': 'ga', 'ITALIAN': 'it', 'JAPANESE': 'ja', 'JAVANESE': 'jw', 'KANNADA': 'kn', 'KAZAKH': 'kk', 'KHMER': 'km', 'KINYARWANDA': 'rw', 'KONKANI': 'gom', 'KOREAN': 'ko', 'KRIO': 'kri', 'KURDISH (KURMANJI)': 'ku', 'KURDISH (SORANI)': 'ckb', 'KYRGYZ': 'ky', 'LAO': 'lo', 'LATIN': 'la', 'LATVIAN': 'lv', 'LINGALA': 'ln', 'LITHUANIAN': 'lt', 'LUGANDA': 'lg', 'LUXEMBOURGISH': 'lb', 'MACEDONIAN': 'mk', 'MAITHILI': 'mai', 'MALAGASY': 'mg', 'MALAY': 'ms', 'MALAYALAM': 'ml', 'MALTESE': 'mt', 'MAORI': 'mi', 'MARATHI': 'mr', 'MEITEILON (MANIPURI)': 'mni-mtei', 'MIZO': 'lus', 'MONGOLIAN': 'mn', 'MYANMAR': 'my', 'NEPALI': 'ne', 'NORWEGIAN': 'no', 'ODIA (ORIYA)': 'or', 'OROMO': 'om', 'PASHTO': 'ps', 'PERSIAN': 'fa', 'POLISH': 'pl', 'PORTUGUESE': 'pt', 'PUNJABI': 'pa', 'QUECHUA': 'qu', 'ROMANIAN': 'ro', 'RUSSIAN': 'ru', 'SAMOAN': 'sm', 'SANSKRIT': 'sa', 'SCOTS GAELIC': 'gd', 'SEPEDI': 'nso', 'SERBIAN': 'sr', 'SESOTHO': 'st', 'SHONA': 'sn', 'SINDHI': 'sd', 'SINHALA': 'si', 'SLOVAK': 'sk', 'SLOVENIAN': 'sl', 'SOMALI': 'so', 'SPANISH': 'es', 'SUNDANESE': 'su', 'SWAHILI': 'sw', 'SWEDISH': 'sv', 'TAJIK': 'tg', 'TAMIL': 'ta', 'TATAR': 'tt', 'TELUGU': 'te', 'THAI': 'th', 'TIGRINYA': 'ti', 'TSONGA': 'ts', 'TURKISH': 'tr', 'TURKMEN': 'tk', 'TWI': 'ak', 'UKRAINIAN': 'uk', 'URDU': 'ur', 'UYGHUR': 'ug', 'UZBEK': 'uz', 'VIETNAMESE': 'vi', 'WELSH': 'cy', 'XHOSA': 'xh', 'YIDDISH': 'yi', 'YORUBA': 'yo', 'ZULU': 'zu'}

# def summarize():
    # summarizer = pipeline("summarization")
    # # Fetch transcript from YouTube video
    # a=(link.find('='))+1
    # c=link[a:a+11]
    # vid_id = c
    # data=yta.get_transcript(vid_id)

    # transcript=''
    # for value in data:
    #     for key,val in value.items():
    #         if key=='text':
    #             transcript+=val + " "

    # l1=transcript.splitlines()
    # global final_tra1
    # final_tra1=" ".join(l1)
    # # Summarize transcript
    # num_iters = int(len(final_tra1)/4000)
    # summarized_text = []
    # for i in range(0, num_iters + 1):
    #     start = 0
    #     start = i * 1000
    #     end = (i + 1) * 1000
    #     print("input text" + final_tra1[start:end])
    #     out = summarizer(final_tra1[start:end])
    #     out = out[0]
    #     out = out['summary_text']
    #     # print("Summarized text"+out)
    #     summarized_text.append(out)
    # summarized_text_str=str(summarized_text)
    # summarized_text_str=summarized_text_str.replace('[', '')
    # summarized_text_str=summarized_text_str.replace(']', '')
    # summarized_text_str=summarized_text_str.replace('"', '')
    # st.write("Summary:")
    # st.write(summarized_text_str)

    # tts = gTTS(summarized_text_str,lang='en')
    # mp3_fp = io.BytesIO()
    # tts.write_to_fp(mp3_fp)
    # st.audio(mp3_fp, format='audio/mp3')

def translate(langs):
    summarized_text_lang = []
    
    for text in summarized_text:
        out = GoogleTranslator(source='auto', target=langs).translate(text)
        summarized_text_lang.append(out)
    summarized_text_lang_str = str(summarized_text_lang)
    summarized_text_lang_str=summarized_text_lang_str.replace('[', '')
    summarized_text_lang_str=summarized_text_lang_str.replace(']', '')
    summarized_text_lang_str=summarized_text_lang_str.replace('"', '')
    st.write("Summary:")
    st.write(summarized_text_lang_str)

    tts = gTTS(summarized_text_lang_str,lang=langs)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)

    # Play audio
    st.audio(mp3_fp, format='audio/mp3')
st.title("YouTube Video Summarizer")
link = st.text_input("Enter the link to the YouTube Video:", value = "https://www.youtube.com/watch?v=gfbSIdwL0FE")

summarizer = pipeline("summarization")
# Fetch transcript from YouTube video
a=(link.find('='))+1
c=link[a:a+11]
vid_id = c
data=yta.get_transcript(vid_id)

transcript=''
for value in data:
    for key,val in value.items():
        if key=='text':
            transcript+=val + " "

l1=transcript.splitlines()
global final_tra1
final_tra1=" ".join(l1)
# Summarize transcript
num_iters = int(len(final_tra1)/4000)
summarized_text = []
for i in range(0, num_iters + 1):
    start = 0
    start = i * 1000
    end = (i + 1) * 1000
    print("input text" + final_tra1[start:end])
    out = summarizer(final_tra1[start:end])
    out = out[0]
    out = out['summary_text']
    # print("Summarized text"+out)
    summarized_text.append(out)
summarized_text_str=str(summarized_text)
summarized_text_str=summarized_text_str.replace('[', '')
summarized_text_str=summarized_text_str.replace(']', '')
summarized_text_str=summarized_text_str.replace('"', '')

# if st.button("Summarize"):
#     summarize()
selected_language = st.selectbox("Select language", LANGS)
if st.button("Summarize"):
    translate(Langs_dict[selected_language])
