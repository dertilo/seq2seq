import os

import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

from seq2seq_chatbot import ChatBot

"""
based on: https://github.com/stephch/rasa_start/blob/master/dash_demo_app.py
"""

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

conv_hist = []
conversation_style = {
    "height": "400px",
    "overflow": "auto",
    "display": "flex",
    "flex-direction": "column-reverse",
}
# fmt: off
app.layout = html.Div([
    html.H3('CoqaBot', style={'marginLeft':'10%'}),
    html.Div([
        html.Div([
            html.Div(id='conversation', style=conversation_style),
            html.Br(),
            html.Table([
                html.Tr([
                    # text input for user message
                    html.Td([dcc.Input(id='msg_input', value='', type='text')],
                            ),
                    # message to send user message to bot backend
                    html.Td([html.Button('Send', id='send_button', type='submit')],
                            )
                ])
            ])],
            style={'width': '325px'}),
    ],
        id='screen',
        className="two columns",
        style={'width': '400px','height': '800px','valign': 'left'}),

    html.Div([html.H5('Background'),
              html.Div(id="background")],
             className="two columns",
             style={'width': '600px'}
             )
],className="row")
# fmt: on


@app.callback(
    [Output(component_id="conversation", component_property="children"),
    Output(component_id="background", component_property="children")],
    [Input(component_id="send_button", component_property="n_clicks")],
    state=[State(component_id="msg_input", component_property="value")],
)
def update_conversation(n_clicks, text):
    global conv_hist

    if n_clicks is not None and n_clicks > 0:
        response,background = chatbot.respond(text)  # agent.handle_message(text)
        user_utt = [html.H5(text, style={"text-align": "right"})]
        bot_utt = [html.H5(html.I(response), style={"text-align": "left"})]
        conv_hist = bot_utt + user_utt + [html.Hr()] + conv_hist
        # conv_hist += [{"user":text}]+[{"bot":response}]
        # conv_table = dash_table.DataTable(
        #     data=conv_hist,
        #     columns=[{'id': c, 'name': c} for c in ["bot", "user"]],
        #     page_action='none',
        #     style_table={'height': '300px', 'overflowY': 'auto'},
        #     style_data = {'border': '0px'},
        #     style_header={
        #         'backgroundColor': 'white',
        #         'fontColor': 'white'
        #     }
        # )
        return conv_hist, background
    else:
        return "",""


@app.callback(
    Output(component_id="msg_input", component_property="value"),
    [Input(component_id="conversation", component_property="children")],
)
def clear_input(_):
    return ""

dummy_background="""In the office of the German Chancellor Angela Merkel, there is a picture of Catherine the Great, the legendary Russian Empress. When asked why she has the picture, Merkel says, "She was a strong woman". Many say the same of Merkel. The most powerful woman in the world, according to US Forbes magazine, was in China last week. She came to discuss trade and environmental issues with China's top leaders. Germany's first woman leader is known as a brave and practical statesman . Even since her time at school, she had the habit of getting everything in order. Every day before doing her homework she would clean the desk and think about what to do next. "I prefer a long time for full preparations to make my decision. But once I decide, I will stand up for what I believe," Merkel said. Perhaps it was good habits that helped her do well in her studies. At 32, she got a doctorate in physics and then she worked as a researcher. However, the life of a scholar couldn't put off her love of politics. While working in labs, Merkel took time off to read political books and at last joined a political party. "Her calmness helped her stand out in the party. She could always find a way out while others felt hopeless," said one of her old friends. In her first big political job as Minister for the Environment in 1994, her scientific background proved very useful. In 2005 she became Germany's youngest chancellor since the second World War. Now half way through her four-year term, the 53-year-old woman has made a name for herself both in Germany and abroad. At the EU summit in 2005 when France quarreled with Britain over the EU budget , some people believed the EU was close to breaking down. But Merkel didn't give up. She shuttled between the heads of the two powers and had them reached an agreement. "Strength comes from composure and courage. Many people say I am a strong woman. But I would rather say I have perseverance," said Merkel."""
class DummyChatBot:
    def respond(self,utt):
        return "bla", dummy_background
# run app
if __name__ == "__main__":
    file = "checkpointepoch=2.ckpt"
    model_file = os.environ["HOME"] + "/data/bart_coqa_seq2seq/" + file

    with ChatBot(model_file) as chatbot:
        app.run_server(debug=True,host="0.0.0.0")
    # chatbot = DummyChatBot()
    # app.run_server(debug=True,host="0.0.0.0")
