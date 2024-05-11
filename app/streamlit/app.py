import streamlit as st
import extra_streamlit_components as stx
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import requests
import graphviz
import datetime
import os
import pickle
import base64

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


# TODO: Make alpha in IOD configurable (server)
# TODO: Make m.max configurable

# launch command
# streamlit run app.py --server.enableXsrfProtection false

#  ,---.   ,--.            ,--.             ,--.        ,--.  ,--.   
# '   .-',-'  '-. ,--,--.,-'  '-. ,---.     |  |,--,--, `--',-'  '-. 
# `.  `-.'-.  .-'' ,-.  |'-.  .-'| .-. :    |  ||      \,--.'-.  .-' 
# .-'    | |  |  \ '-'  |  |  |  \   --.    |  ||  ||  ||  |  |  |   
# `-----'  `--'   `--`--'  `--'   `----'    `--'`--''--'`--'  `--' 

if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'server_url' not in st.session_state:
    if 'LITESTAR_HOST' in os.environ and 'LITESTAR_PORT' in os.environ:
        st.session_state['server_url'] = f'http://{os.environ["LITESTAR_HOST"]}:{os.environ["LITESTAR_PORT"]}' # TODO: load default url from config file
    else:
        st.session_state['server_url'] = 'http://127.0.0.1:8000'
if 'last_health_check' not in st.session_state:
    st.session_state['last_health_check'] = None
if 'is_connected_to_server' not in st.session_state:
    st.session_state['is_connected_to_server'] = None
    
if 'server_provided_user_id' not in st.session_state:
    st.session_state['server_provided_user_id'] = None
if 'current_room' not in st.session_state:
    st.session_state['current_room'] = None
    
if '_alpha_value' not in st.session_state:
    st.session_state['_alpha_value'] = 0.05

if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'result_pvals' not in st.session_state:
    st.session_state['result_pvals'] = None
if 'result_labels' not in st.session_state:
    st.session_state['result_labels'] = None
if 'server_has_received_data' not in st.session_state:
    st.session_state['server_has_received_data'] = False  
    
# ,--.  ,--.,--------.,--------.,------.     ,------.                ,--.      ,--.  ,--.       ,--.                      
# |  '--'  |'--.  .--''--.  .--'|  .--. '    |  .--. ' ,---.  ,---.,-'  '-.    |  '--'  | ,---. |  | ,---.  ,---. ,--.--. 
# |  .--.  |   |  |      |  |   |  '--' |    |  '--' || .-. |(  .-''-.  .-'    |  .--.  || .-. :|  || .-. || .-. :|  .--' 
# |  |  |  |   |  |      |  |   |  | --'     |  | --' ' '-' '.-'  `) |  |      |  |  |  |\   --.|  || '-' '\   --.|  |    
# `--'  `--'   `--'      `--'   `--'         `--'      `---' `----'  `--'      `--'  `--' `----'`--'|  |-'  `----'`--'    
#                                                                                                   `--'  
    
def post_to_server(url, payload):
    try:
        r = requests.post(url=url, json=payload)
        return r
    except:
        st.session_state['last_health_check'] = None
        st.error('There are problems with the server connection')
    return None
        

#  ,---.                                           ,--.  ,--.               ,--.  ,--.  ,--.             ,-----.,--.                  ,--.     
# '   .-'  ,---. ,--.--.,--.  ,--.,---. ,--.--.    |  '--'  | ,---.  ,--,--.|  |,-'  '-.|  ,---. ,-----.'  .--./|  ,---.  ,---.  ,---.|  |,-.  
# `.  `-. | .-. :|  .--' \  `'  /| .-. :|  .--'    |  .--.  || .-. :' ,-.  ||  |'-.  .-'|  .-.  |'-----'|  |    |  .-.  || .-. :| .--'|     /  
# .-'    |\   --.|  |     \    / \   --.|  |       |  |  |  |\   --.\ '-'  ||  |  |  |  |  | |  |       '  '--'\|  | |  |\   --.\ `--.|  \  \  
# `-----'  `----'`--'      `--'   `----'`--'       `--'  `--' `----' `--`--'`--'  `--'  `--' `--'        `-----'`--' `--' `----' `---'`--'`--'
    
def check_server_connection():
    curr_time = datetime.datetime.now()
    if st.session_state['last_health_check'] is not None and (curr_time - st.session_state['last_health_check']).total_seconds() < 60*5:
        return True
    try:
        with st.spinner('Checking connection to server...'):
            r = requests.get(url = f'{st.session_state["server_url"]}/health-check')
            
        if r.status_code != 200:
            st.error('There are problems with the server connection')
            st.session_state['is_connected_to_server'] = False
            return False
    except:
        st.error('There are problems with the server connection')
        st.session_state['is_connected_to_server'] = False
        return False
    
    st.session_state['last_health_check'] = curr_time
    st.session_state['is_connected_to_server'] = True
    return True

#  ,---.                                            ,-----.,--.                  ,--.          ,--.         
# '   .-'  ,---. ,--.--.,--.  ,--.,---. ,--.--.    '  .--./|  ,---.  ,---.  ,---.|  |,-.,-----.|  |,--,--,  
# `.  `-. | .-. :|  .--' \  `'  /| .-. :|  .--'    |  |    |  .-.  || .-. :| .--'|     /'-----'|  ||      \ 
# .-'    |\   --.|  |     \    / \   --.|  |       '  '--'\|  | |  |\   --.\ `--.|  \  \       |  ||  ||  | 
# `-----'  `----'`--'      `--'   `----'`--'        `-----'`--' `--' `----' `---'`--'`--'      `--'`--''--' 

def step_check_in_to_server():
    if st.session_state['server_provided_user_id'] is not None:
        st.info('Server check-in completed')
        
    st.write('Please enter the server URL:')
    col1, col2 = st.columns((6,1))
    server_url = col1.text_input('Please chose your username', placeholder=st.session_state['server_url'], label_visibility='collapsed')
    
    if col2.button(':link:', help='Connect to URL') and server_url is not None and server_url != '':
        if st.session_state['username'] is not None:
            st.warning('''You are already checked in with a server.  
                       In order to leave the server, refresh this page.''')
        else:
            if server_url.endswith('/'):
                server_url = server_url[:-1]
            st.session_state['server_url'] = server_url
            st.session_state['last_health_check'] = None
            st.rerun()
            return
        
    st.write('Please enter a username:')
    col1, col2 = st.columns((6,1))
    username = col1.text_input('Please chose your username', placeholder=st.session_state['username'], label_visibility='collapsed')
    
    if st.session_state['server_provided_user_id'] is None:
        button = col2.button(':arrow_heading_up:', help='Submit check-in request')
        request_url = f'{st.session_state["server_url"]}/check-in'
        request_params = {'username': username}
    else:
        button = col2.button(':arrow_heading_up:', help='Change name', disabled=not username)
        request_url = f'{st.session_state["server_url"]}/change-name'
        request_params = {
            'id': st.session_state['server_provided_user_id'],
            'username': st.session_state['username'],
            'new_username': username
            }

    if button:
        r = post_to_server(request_url, request_params)
        if r is None:
            return
        if r.status_code != 200:
            st.error('Failed to check in with the server. Please reload the page')
            return

        r = r.json()
        st.session_state['server_provided_user_id'] = r['id']
        st.session_state['username'] = r['username']
        st.rerun()
    return

# ,------.            ,--.              ,--. ,--.       ,--.                  ,--. 
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  | |  | ,---. |  | ,---.  ,--,--. ,-|  | 
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  | |  || .-. ||  || .-. |' ,-.  |' .-. | 
# |  '--'  /\ '-'  |  |  |  \ '-'  |    '  '-'  '| '-' '|  |' '-' '\ '-'  |\ `-' | 
# `-------'  `--`--'  `--'   `--`--'     `-----' |  |-' `--' `---'  `--`--' `---'  
#                                                `--'    

def step_upload_data():
    uploaded_file = st.file_uploader('Data Upload', type='csv')
            
    # read uploaded file into session state
    if uploaded_file is not None:
        st.session_state['uploaded_data'] = pd.read_csv(uploaded_file)
        # Reset when new file is uploaded
        st.session_state['result_pvals'] = None
        st.session_state['result_labels'] = None
    # if uploaded data exists in session state, make available to show
    if st.session_state['uploaded_data'] is not None:
        with st.expander('View Data'):
            df = st.session_state['uploaded_data']
            st.dataframe(dataframe_explorer(df), use_container_width=True)
            
# ,------.            ,--.              ,------.                                          ,--.                
# |  .-.  \  ,--,--.,-'  '-. ,--,--.    |  .--. ',--.--. ,---.  ,---. ,---.  ,---.  ,---. `--',--,--,  ,---.  
# |  |  \  :' ,-.  |'-.  .-'' ,-.  |    |  '--' ||  .--'| .-. || .--'| .-. :(  .-' (  .-' ,--.|      \| .-. | 
# |  '--'  /\ '-'  |  |  |  \ '-'  |    |  | --' |  |   ' '-' '\ `--.\   --..-'  `).-'  `)|  ||  ||  |' '-' ' 
# `-------'  `--`--'  `--'   `--`--'    `--'     `--'    `---'  `---' `----'`----' `----' `--'`--''--'.`-  /  
#                                                                                                     `---'  

def step_process_data():
    _, col1, col2, _ = st.columns((1,1,1,1))
    if st.session_state['result_pvals'] is None:
        button = col1.button("Process Data!")
    else:
        button = col1.button("Reprocess Data!")
    
    if button:
        with st.spinner('Data is being processed...'):
            # Read data from state
            df = st.session_state['uploaded_data']
            
            # Call R function
            with (ro.default_converter + pandas2ri.converter).context():
                # load local-ci script
                ro.r['source']('./scripts/local-ci.r')
                # load function from R script
                run_ci_test_f = ro.globalenv['run_ci_test']
                
                #converting it into r object for passing into r function
                df_r = ro.conversion.get_conversion().py2rpy(df)
                #Invoking the R function and getting the result
                result = run_ci_test_f(df_r)
                #Converting it back to a pandas dataframe.
                df_pvals = ro.conversion.get_conversion().rpy2py(result['citestResults'])
                labels = list(result['labels'])
            
        st.session_state['result_pvals'] = df_pvals
        st.session_state['result_labels'] = labels
        st.rerun()
        
    if col2.button('Submit Data!', help='Submit Data to Server', disabled=st.session_state['result_pvals'] is None):
        df_pvals = st.session_state['result_pvals']
        labels = st.session_state['result_labels']
        base64_df = base64.b64encode(pickle.dumps(df_pvals)).decode('utf-8')
        # send data and labels
        r = post_to_server(url = f'{st.session_state["server_url"]}/submit-data', payload={'id': st.session_state['server_provided_user_id'],
                                                                    'username': st.session_state['username'],
                                                                    'data': base64_df,
                                                                    'data_labels': labels})
        if r is None:
            return
        if r.status_code != 200:
            st.error('An error occured when submitting the data')
            return
        st.session_state['server_has_received_data'] = True
    
    if st.session_state['server_has_received_data'] == True:
        st.info('The server has received your data')
        
    if st.session_state['result_pvals'] is not None and st.session_state['server_has_received_data'] == False:
        # TODO as dialog?
        st.warning('''Any submitted data can be accessed by the server.  
                                 Any participants in the same room will be able to access the data labels.  
                                 Once you join a room, your data can be used in the processing rooms data.  
                                 Be sure no sensitive data is submitted!''')

    if st.session_state['result_pvals'] is not None:
        df_pvals = st.session_state['result_pvals'].copy()
        labels = st.session_state['result_labels']
        #with st.expander('Variable Mapping'):
        #    for id2label in [f'{i+1} $\\rightarrow$ {l}' for i,l in enumerate(labels)]:
        #        st.markdown("- " + id2label)
        df_pvals['X'] = df_pvals['X'].apply(lambda x: labels[int(x)-1])
        df_pvals['Y'] = df_pvals['Y'].apply(lambda x: labels[int(x)-1])
        df_pvals['S'] = df_pvals['S'].apply(lambda x: ','.join(sorted([labels[int(xi)-1] for xi in x.split(',') if xi != ''])))
        st.dataframe(dataframe_explorer(df_pvals), use_container_width=True)
        
    return

# ,------.                             ,--.          ,--.   ,--.            
# |  .--. ' ,---.  ,---. ,--,--,--.    |  |    ,---. |  |-. |  |-.,--. ,--. 
# |  '--'.'| .-. || .-. ||        |    |  |   | .-. || .-. '| .-. '\  '  /  
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--.' '-' '| `-' || `-' | \   '   
# `--' '--' `---'  `---' `--`--`--'    `-----' `---'  `---'  `---'.-'  /    
#                                                                 `---'   

def step_join_rooms():
    # enter new room name and join or create it
    info_placeholder = st.empty()
    st.write('Please enter a room name:')
    col1, col2, col3, col4 = st.columns((7,1,1,1))
    # TODO: refresh button
    room_name = col1.text_input('Please chose a room name', label_visibility='collapsed')
    if col2.button(':arrow_right:', help='Join room', disabled=room_name is None):
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room_name}/join', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return 
        st.error('An error occured trying to join the room')
    
    if col3.button(':tada:', help='Create room', disabled=room_name is None):
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/create', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username'],
                                                                'room_name': room_name})
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error('An error occured trying to create the room')
        
    if col4.button(':arrows_counterclockwise:', help='Refresh the room'):
        st.rerun()
        return

    # Get room list
    r = post_to_server(url = f'{st.session_state["server_url"]}/rooms', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
    if r is None:
        return
    if r.status_code != 200:
        st.error('An error occured trying to fetch room data')
        return
    
    rooms = r.json()
    
    if len(rooms) == 0:
        info_placeholder.info('There are no rooms yet, but you may create your own!')
    
    col_structure = (3,3,1)
    room_fields = ['Name', 'Owner', 'Join']
    
    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)
    for i, room in enumerate(rooms):
        col1, col2, col3 = st.columns(col_structure)
        col1.write(room['name'])
        col2.write(room['owner_name'])
        if col3.button(':arrow_right:', help='Room is locked' if room['is_locked'] else 'Join', disabled=room['is_locked'], key=f'join-button-{i}'):
            r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/join', payload={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
            if r is None:
                return
            if r.status_code == 200:
                st.session_state['current_room'] = r.json()
                st.rerun()
                return
            
            st.error('An error occured trying to join the room')
    return

# ,------.                             ,------.           ,--.          ,--.,--.        
# |  .--. ' ,---.  ,---. ,--,--,--.    |  .-.  \  ,---. ,-'  '-. ,--,--.`--'|  | ,---.  
# |  '--'.'| .-. || .-. ||        |    |  |  \  :| .-. :'-.  .-'' ,-.  |,--.|  |(  .-'  
# |  |\  \ ' '-' '' '-' '|  |  |  |    |  '--'  /\   --.  |  |  \ '-'  ||  ||  |.-'  `) 
# `--' '--' `---'  `---' `--`--`--'    `-------'  `----'  `--'   `--`--'`--'`--'`----' 

def step_show_room_details():
    room = st.session_state['current_room']
    if room['is_processing']:
        st.write(f"## Room: {st.session_state['current_room']['name']} (in progress)")
    elif room['is_finished']:
        st.write(f"## Room: {st.session_state['current_room']['name']} (finished)")
    else:
        st.write(f"## Room: {st.session_state['current_room']['name']} ({'hidden' if room['is_hidden'] else 'public'}) ({'locked' if room['is_locked'] else 'open'})")
    
    _, col1, col2, col3, col4, col5, _ = st.columns((1,1,1,1,1,1,1))
        
    if col1.button(':arrows_counterclockwise:', help='Refresh the room'):
        st.rerun()
        return
    
    if room['is_locked']:
        lock_button_text = ':lock:'
        lock_button_help_text = 'Unlock the room'
    else:
        lock_button_text = ':unlock:'
        lock_button_help_text = 'Lock the room'
    
    if col2.button(lock_button_text, help=lock_button_help_text, disabled=st.session_state['username']!=room['owner_name']):
        lock_endpoint = 'unlock' if room['is_locked'] else 'lock'
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/{lock_endpoint}', payload={'id': st.session_state['server_provided_user_id'],
                                                                                             'username': st.session_state['username']})
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to {lock_endpoint} the room')
        
    if room['is_hidden']:
        hide_button_text = ':face_in_clouds:'
        hide_button_help_text = 'Reveal the room'
    else:
        hide_button_text = ':eyes:'
        hide_button_help_text = 'Hide the room'
        
    if col3.button(hide_button_text, help=hide_button_help_text, disabled=st.session_state['username']!=room['owner_name']):
        hide_endpoint = 'reveal' if room['is_hidden'] else 'hide'
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/{hide_endpoint}', payload={'id': st.session_state['server_provided_user_id'],
                                                                                             'username': st.session_state['username']})
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to {hide_endpoint} the room')
        
    if col4.button(':arrow_left:', help='Leave the room'):
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/leave', payload={'id': st.session_state['server_provided_user_id'],
                                                                                 'username': st.session_state['username']})
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = None
            st.rerun()
            return
        st.error(f'An error occured while trying to leave the room')
        
    if col5.button(':fire:', help='Run IOD on participant data', disabled=st.session_state['username']!=room['owner_name']):
        r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/run', payload={'id': st.session_state['server_provided_user_id'],
                                                                                 'username': st.session_state['username'],
                                                                                 'alpha': round(st.session_state['alpha_value'],2)
                                                                                 })
        if r is None:
            return
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to run IOD')
        
    def change_alpha_value():
        st.session_state['_alpha_value'] = st.session_state['alpha_value']
        
    # Run config
    _, col1, _ = st.columns((1,5,1))
    col1.number_input('Select alpha value:',
                      value=st.session_state['_alpha_value'],
                      min_value=0.0,
                      max_value=1.0,
                      step=0.01,
                      key='alpha_value',
                      format='%.2f',
                      on_change=change_alpha_value)
    
    col_structure = (1,3,3,1)
    room_fields = ['№', 'Name', 'Provided Labels', 'Action']
    cols = st.columns(col_structure)
    for col, field_name in zip(cols, room_fields):
        col.write(field_name)

    for i, user in enumerate(room['users']):
        col1, col2, col3, col4 = st.columns(col_structure)
        col1.write(i)
        
        user_str = f"{user}"
        if user == st.session_state['username']:
            user_str += " *(you)*"
        if user == room['owner_name']:
            user_str += " *(owner)*"
        col2.write(user_str)
        
        with col3.expander('Show'):
            for label in sorted(room['user_provided_labels'][user]):
                st.markdown(f'- {label}')
        
        if user != st.session_state['username']:
            if col4.button(':x:', help='Kick', disabled=st.session_state['username'] != room['owner_name'], key=f'kick-button-{i}'):
                r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{room["name"]}/kick/{user}', payload={'id': st.session_state['server_provided_user_id'],
                                                                                                                'username': st.session_state['username']})
                if r is None:
                    return
                if r.status_code == 200:
                    st.session_state['current_room'] = r.json()
                    st.rerun()
                    return
                st.error(f'Failed to kick user')
    return

# ,------.                       ,--.  ,--.      ,--.   ,--.,--.                       ,--.,--.                 ,--.  ,--.                
# |  .--. ' ,---.  ,---. ,--.,--.|  |,-'  '-.     \  `.'  / `--' ,---. ,--.,--. ,--,--.|  |`--',-----. ,--,--.,-'  '-.`--' ,---. ,--,--,  
# |  '--'.'| .-. :(  .-' |  ||  ||  |'-.  .-'      \     /  ,--.(  .-' |  ||  |' ,-.  ||  |,--.`-.  / ' ,-.  |'-.  .-',--.| .-. ||      \ 
# |  |\  \ \   --..-'  `)'  ''  '|  |  |  |         \   /   |  |.-'  `)'  ''  '\ '-'  ||  ||  | /  `-.\ '-'  |  |  |  |  |' '-' '|  ||  | 
# `--' '--' `----'`----'  `----' `--'  `--'          `-'    `--'`----'  `----'  `--`--'`--'`--'`-----' `--`--'  `--'  `--' `---' `--''--' 

arrow_type_lookup = {
        1: 'odot',
        2: 'normal',
        3: 'none'
    }
def data2graph(data, labels):
    graph = graphviz.Digraph()
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            arrhead = data[i][j]
            arrtail = data[j][i]
            if data[i][j] == 1:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 2:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
            elif data[i][j] == 3:
                graph.edge(labels[i], labels[j], arrowtail=arrow_type_lookup[arrtail], arrowhead=arrow_type_lookup[arrhead])
                
    return graph

def step_view_results():
    room = st.session_state['current_room']
    result_graphs = [data2graph(d,l) for d,l in zip(room['result'], room['result_labels'])]
    private_result_graph = data2graph(room['private_result'], room['private_labels'])
    
    t1, t2 = st.tabs(['Combined Results', 'Private Result'])
    
    with t1:
        cols = st.columns((1,1,1))
        for i,g in enumerate(result_graphs):
            cols[i%3].graphviz_chart(g)
            cols[i%3].write('---')
            
    with t2:
        _, col1, _ = st.columns((1,1,1))
        col1.graphviz_chart(private_result_graph)
        
    # use visualization library to show images of pags as well
    return

# ,--.   ,--.        ,--.            ,------.                            ,---.   ,--.                         ,--.                         
# |   `.'   | ,--,--.`--',--,--,     |  .--. ' ,--,--. ,---.  ,---.     '   .-',-'  '-.,--.--.,--.,--. ,---.,-'  '-.,--.,--.,--.--. ,---.  
# |  |'.'|  |' ,-.  |,--.|      \    |  '--' |' ,-.  || .-. || .-. :    `.  `-.'-.  .-'|  .--'|  ||  || .--''-.  .-'|  ||  ||  .--'| .-. : 
# |  |   |  |\ '-'  ||  ||  ||  |    |  | --' \ '-'  |' '-' '\   --.    .-'    | |  |  |  |   '  ''  '\ `--.  |  |  '  ''  '|  |   \   --. 
# `--'   `--' `--`--'`--'`--''--'    `--'      `--`--'.`-  /  `----'    `-----'  `--'  `--'    `----'  `---'  `--'   `----' `--'    `----'

def main():
    
    st.write('# Welcome to {Some App}')

    col1, col2, _ = st.columns((1,1,3))
    col1.write('<sup>View our paper [here](https://www.google.com)</sup>', unsafe_allow_html=True)
    col2.write('<sup>View our GitHub [here](https://www.google.com)</sup>', unsafe_allow_html=True)
    
    if check_server_connection():
        st.info('Server connection established' + ('' if st.session_state['username'] is None else f" - checked in as: {st.session_state['username']}"))
  
    refresh_failure = False
    # refresh current room
    if st.session_state['current_room'] is not None:
        try:
            r = post_to_server(url = f'{st.session_state["server_url"]}/rooms/{st.session_state["current_room"]["name"]}', payload={'id': st.session_state['server_provided_user_id'],
                                                                                                            'username': st.session_state['username']})
            if r is None:
                return
            if r.status_code == 200:
                st.session_state['current_room'] = r.json()
            else:
                st.session_state['current_room'] = None
        except:
            refresh_failure = True
        
    step = stx.stepper_bar(steps=["Server Check-In", "Upload Data", "Process Data", "Join Room", "View Result"], lock_sequence=False)
    
    if refresh_failure:
        st.error('An error occured trying to update current room data')
        return
    
    if step > 0 and st.session_state['is_connected_to_server'] == False:
        st.warning('Please ensure you have a connection to the server before continuing')
        return

    if step == 0:
        # todo: add ip field for server
        step_check_in_to_server()
    elif step == 1:
        if st.session_state['username'] is None:
            st.info("Please check in with the server before continuing")
            return
        step_upload_data()
    elif step == 2:
        if st.session_state['uploaded_data'] is None:
            st.info("Please upload a file before continuing")
            return
        step_process_data()
    elif step == 3:
        if st.session_state['server_has_received_data'] is False:
            st.info("Please send your data to the server before continuing")
            return
        if st.session_state['current_room'] is None:
            step_join_rooms()
        else:
            step_show_room_details()
    elif step == 4:
        # TODO: verify that data has provided results
        if st.session_state['current_room'] is None or st.session_state['current_room']['result'] is None:
            st.info("Please run IOD before attempting to view results")
            return
        step_view_results()
    else:
        st.error('An error occured. Please reload the page')

    if step > 2:
        st_autorefresh(interval=3000, limit=100, key="autorefresh")
        
main()