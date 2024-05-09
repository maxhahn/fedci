import streamlit as st
import extra_streamlit_components as stx
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd
import requests
import pickle
import base64
import graphviz

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

server_url = 'http://127.0.0.1:8000'

# streamlit run app.py --server.enableXsrfProtection false

# create personal secret / personal uid
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'stepper_state' not in st.session_state:
    st.session_state['stepper_state'] = None
if 'server_url' not in st.session_state:
    st.session_state['server_url'] = 'http://127.0.0.1:8000'
    
# set server_provided_user_id when doing initial check up with server
if 'server_provided_user_id' not in st.session_state:
    st.session_state['server_provided_user_id'] = None
if 'current_room' not in st.session_state:
    st.session_state['current_room'] = None

if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'result_pvals' not in st.session_state:
    st.session_state['result_pvals'] = None
if 'result_labels' not in st.session_state:
    st.session_state['result_labels'] = None
if 'server_has_received_data' not in st.session_state:
    st.session_state['server_has_received_data'] = False  
    
    
def check_server_connection():
    with st.spinner('Checking connection to server...'):
        r = requests.get(url = f'{server_url}/health-check')
    status_code = r.status_code
    
    if status_code != 200:
        st.error('There are problems with the server connection')
        print('SERVER CONNECTION ISSUES')
        print(r)
        return
    
    st.info('Server connection established' + ('' if st.session_state['username'] is None else f" - checked in as: {st.session_state['username']}"))
    return
    
def show_room_sidebar():
    
    col_structure = (1,4,2)
    partner_table_fields = ["№", 'user', 'action']
    
    # TODO: Refresh and Lock Buttons to stop other people from joining
    
    client_is_room_owner = st.session_state['server_provided_user_id'] == st.session_state['current_room_owner']
    
    with st.sidebar:
        # room title
        st.write(f"Room: {st.session_state['current_room']} ({'locked' if st.session_state['current_room_is_locked'] else 'open'})")
        
        # room lock and refresh
        if client_is_room_owner:
            _, col1, col2, _ = st.columns((1,1,1,1))
        else:
            _, col1, _ = st.columns((1,1,1))
            
        if col1.button(':arrows_counterclockwise:', help='Refresh the room'):
            st.rerun()
        else:
            # if the button wasn't pressed, stepper can be moved. If it was pressed. Stay on the same stepper state
            st.session_state['stepper_state'] = None
        st.write(f"Room: {st.session_state['current_room']} ({'locked' if st.session_state['current_room_is_locked'] else 'open'})")
        
        if client_is_room_owner:
            if st.session_state['current_room_is_locked']:
                lock_button_text = ':lock:'
                lock_button_help_text = 'Unlock the room'
            else:
                lock_button_text = ':unlock:'
                lock_button_help_text = 'Lock the room'
            if col2.button(lock_button_text, help=lock_button_help_text):
                st.session_state['current_room_is_locked'] = not st.session_state['current_room_is_locked']
                # TODO: Post to server
                st.rerun()
        
        # room member list
        cols = st.columns(col_structure)
        for col, field_name in zip(cols, partner_table_fields):
            col.write(field_name)
            
        users = st.session_state['current_partners']
            
        for i, user in enumerate(users):
            col1, col2, col3 = st.columns(col_structure)
            col1.write(i)
            user_str = "{}"
            if user == st.session_state['server_provided_user_id']:
                user_str += " (you)"
            if user == st.session_state['current_room_owner']:
                user_str += " (owner)"

            col2.write(user_str.format(user))
            if client_is_room_owner:
                if user == st.session_state['username']:
                    do_action = False
                else: 
                    do_action = col3.button(':x:', key=f"user-kick-button-{i}", help='Kick')
                if do_action:
                    del st.session_state['current_partners'][i]
                    # TODO: post kick to server and just refresh partner ids from response
                    st.rerun()        
        
    return    

def step_check_in_to_server():
    if st.session_state['server_provided_user_id'] is not None:
        st.info('Server check-in completed')
        
    st.write('Please enter a username:')
        
    col1, col2 = st.columns((6,1))
    username = col1.text_input('Please chose your username', placeholder=st.session_state['username'], label_visibility='collapsed')
    
    if st.session_state['server_provided_user_id'] is None:
        button = col2.button(':arrow_heading_up:', help='Submit check-in request')
        request_url = f'{server_url}/check-in'
        request_params = {'username': username}
    else:
        button = col2.button(':arrow_heading_up:', help='Change name', disabled=not username)
        request_url = f'{server_url}/change-name'
        request_params = {
            'id': st.session_state['server_provided_user_id'],
            'username': st.session_state['username'],
            'new_username': username
            }

    if button:
        r = requests.post(url=request_url, json=request_params)
        if r.status_code != 200:
            st.error('Failed to check in with the server. Please reload the page')
            print(r)
            return

        r = r.json()
        st.session_state['server_provided_user_id'] = r['id']
        st.session_state['username'] = r['username']
        st.rerun()
    return

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
                ro.r['source']('../scripts/local-ci.r')
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
        r = requests.post(url = f'{server_url}/submit-data', json={'id': st.session_state['server_provided_user_id'],
                                                                    'username': st.session_state['username'],
                                                                    'data': base64_df,
                                                                    'data_labels': labels})
        
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
        df_pvals = st.session_state['result_pvals']
        labels = st.session_state['result_labels']
        with st.expander('Variable Mapping'):
            for id2label in [f'{i+1} $\\rightarrow$ {l}' for i,l in enumerate(labels)]:
                st.markdown("- " + id2label)
        st.dataframe(dataframe_explorer(df_pvals), use_container_width=True)
        
    return

def step_join_rooms():
    # enter new room name and join or create it
    info_placeholder = st.empty()
    st.write('Please enter a room name:')
    col1, col2, col3, col4 = st.columns((7,1,1,1))
    # TODO: refresh button
    room_name = col1.text_input('Please chose a room name', label_visibility='collapsed')
    if col2.button(':arrow_right:', help='Join room', disabled=room_name is None):
        r = requests.post(url = f'{server_url}/rooms/{room_name}/join', json={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return 
            
        st.error('An error occured trying to join the room')
    
    if col3.button(':tada:', help='Create room', disabled=room_name is None):
        r = requests.post(url = f'{server_url}/rooms/create', json={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username'],
                                                                'room_name': room_name})
        if r.status_code != 200:
            st.error('An error occured trying to create the room')
            return
        
        st.session_state['current_room'] = r.json()
        st.rerun()
        return
    
    if col4.button(':arrows_counterclockwise:', help='Refresh the room'):
        st.rerun()
        return
        
    
    # Get room list
    r = requests.post(url = f'{server_url}/rooms', json={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
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
            r = requests.post(url = f'{server_url}/rooms/{room["name"]}/join', json={'id': st.session_state['server_provided_user_id'],
                                                                'username': st.session_state['username']})
            if r.status_code == 200:
                st.session_state['current_room'] = r.json()
                st.rerun()
                return
            
            st.error('An error occured trying to join the room')
            
    # TODO: create room list
    return

def step_show_room_details():
    room = st.session_state['current_room']
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
        r = requests.post(url = f'{server_url}/rooms/{room["name"]}/{lock_endpoint}', json={'id': st.session_state['server_provided_user_id'],
                                                                                             'username': st.session_state['username']})
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
        r = requests.post(url = f'{server_url}/rooms/{room["name"]}/{hide_endpoint}', json={'id': st.session_state['server_provided_user_id'],
                                                                                             'username': st.session_state['username']})
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
            st.rerun()
            return
        st.error(f'An error occured while trying to {hide_endpoint} the room')
        
    if col4.button(':arrow_left:', help='Leave the room'):
        r = requests.post(url = f'{server_url}/rooms/{room["name"]}/leave', json={'id': st.session_state['server_provided_user_id'],
                                                                                 'username': st.session_state['username']})
        
        if r.status_code == 200:
            st.session_state['current_room'] = None
            st.session_state['room_result'] = None
            st.session_state['room_result_labels'] = None
            st.session_state['private_result'] = None
            st.session_state['private_result_labels'] = None
            st.rerun()
            return
        st.error(f'An error occured while trying to leave the room')
        
        
    if col5.button(':fire:', help='Run IOD on participant data', disabled=st.session_state['username']!=room['owner_name']):
        r = requests.post(url = f'{server_url}/rooms/{room["name"]}/run', json={'id': st.session_state['server_provided_user_id'],
                                                                                 'username': st.session_state['username']})
        
        if r.status_code == 200:
            r = r.json()
            st.session_state['room_result'] = r['result']
            st.session_state['room_result_labels'] = r['result_labels']
            st.session_state['private_result'] = r['private_result']
            st.session_state['private_result_labels'] = r['private_labels']
            st.rerun()
            return
        st.error(f'An error occured while trying to run IOD')
    
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
            user_str += " (you)"
        if user == room['owner_name']:
            user_str += " (owner)"
        col2.write(user_str)
        
        with col3.expander('Show'):
            for label in sorted(room['user_provided_labels'][user]):
                st.markdown(f'- {label}')
        
        if user != st.session_state['username']:
            if col4.button(':x:', help='Kick', disabled=st.session_state['username'] != room['owner_name']):
                # TODO: send kick
                st.rerun()
    
    
    return


arrow_type_lookup = {
        1: 'odot',
        2: 'none',
        3: 'normal'
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
    result_graphs = [data2graph(d,l) for d,l in zip(st.session_state['room_result'], st.session_state['room_result_labels'])]
    private_result_graph = data2graph(st.session_state['private_result'], st.session_state['private_result_labels'])
    
    t1, t2 = st.tabs(['Combined Results', 'Private Results'])
    
    with t1:
        cols = st.columns((1,1,1))
        for i,g in enumerate(result_graphs):
            cols[i%3].graphviz_chart(g)
            cols[i%3].write('---')
            
    with t2:
        st.graphviz_chart(private_result_graph)
        
    # use visualization library to show images of pags as well
    return


def main():
    st.write('# Welcome to {Some App}')
    
    # todo: add time check to only health check periodically. Or only once?
    check_server_connection()
    # TODO: refresh current room when in room alreadz
    if st.session_state['current_room'] is not None:
        r = requests.post(url = f'{server_url}/rooms/{st.session_state["current_room"]["name"]}', json={'id': st.session_state['server_provided_user_id'],
                                                                                                        'username': st.session_state['username']})
        if r.status_code == 200:
            st.session_state['current_room'] = r.json()
        else:
            st.error('An error occured trying to update current room data')
        
    step = stx.stepper_bar(steps=["Server Check-In", "Upload Data", "Process Data", "Join Room", "View Result"], lock_sequence=False)

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
        if st.session_state['room_result'] is None:
            st.info("Please run IOD before attempting to view results")
            return
        step_view_results()
    else:
        st.error('Please reload the page - Stepper out of range')

        
main()