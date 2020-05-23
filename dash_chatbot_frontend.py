import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

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
    html.H3('Testing Bot', style={'text-align': 'center'}),
    html.Div([
        html.Div(id='conversation', style=conversation_style),
        html.Br(),
        html.Div([
            html.Table([
                html.Tr([
                    # text input for user message
                    html.Td([dcc.Input(id='msg_input', value='hello', type='text')],
                            style={'valign': 'middle'}),
                    # message to send user message to bot backend
                    html.Td([html.Button('Send', id='send_button', type='submit')],
                            style={'valign': 'middle'})
                ])
            ])],
            style={'width': '325px', 'margin': '0 auto'}),
    ],
        id='screen',
        style={'width': '400px', 'margin': '0 auto'})
])
# fmt: on


@app.callback(
    Output(component_id="conversation", component_property="children"),
    [Input(component_id="send_button", component_property="n_clicks")],
    state=[State(component_id="msg_input", component_property="value")],
)
def update_conversation(n_clicks, text):
    global conv_hist

    if n_clicks is not None and n_clicks > 0:
        response = "whaat?"  # agent.handle_message(text)
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
        return conv_hist
    else:
        return ""


@app.callback(
    Output(component_id="msg_input", component_property="value"),
    [Input(component_id="conversation", component_property="children")],
)
def clear_input(_):
    return ""


# run app
if __name__ == "__main__":
    app.run_server(debug=True)
