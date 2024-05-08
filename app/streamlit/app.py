import streamlit as st
import extra_streamlit_components as stx
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd
import uuid

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# streamlit run app.py --server.enableXsrfProtection false

# create personal secret / personal uid
if 'private_user_id' not in st.session_state:
    st.session_state['private_user_id'] = uuid.uuid4()
if 'stepper_state' not in st.session_state:
    st.session_state['stepper_state'] = None
    
# set server_provided_user_id when doing initial check up with server
if 'server_provided_user_id' not in st.session_state:
    st.session_state['server_provided_user_id'] = "User A"
if 'current_room' not in st.session_state:
    st.session_state['current_room'] = "My Room"
if 'current_room_is_locked' not in st.session_state:
    st.session_state['current_room_is_locked'] = True
if 'current_room_id' not in st.session_state:
    st.session_state['current_room_id'] = None
if 'current_room_owner' not in st.session_state:
    st.session_state['current_room_owner'] = "User A"
if 'current_partners' not in st.session_state:
    st.session_state['current_partners'] = ['User A', 'User B']
if 'current_partner_ids' not in st.session_state:
    st.session_state['current_partner_ids'] = ['User A', 'User B']

if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'result_pvals' not in st.session_state:
    st.session_state['result_pvals'] = None
if 'result_labels' not in st.session_state:
    st.session_state['result_labels'] = None
    
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
        user_ids = st.session_state['current_partner_ids']
            
        for i, (user, user_id) in enumerate(zip(users, user_ids)):
            col1, col2, col3 = st.columns(col_structure)
            col1.write(i)
            user_str = "{}"
            if user_id == st.session_state['server_provided_user_id']:
                user_str += " (you)"
            if user_id == st.session_state['current_room_owner']:
                user_str += " (owner)"

            col2.write(user_str.format(user))
            if client_is_room_owner:
                if user_id == st.session_state['server_provided_user_id']:
                    do_action = False
                else: 
                    do_action = col3.button(':x:', key=f"user-kick-button-{i}", help='Kick')
                if do_action:
                    del st.session_state['current_partners'][i]
                    del st.session_state['current_partner_ids'][i]
                    # TODO: post kick to server and just refresh partner ids from response
                    st.rerun()        
        
    return    

def step_connect_to_server():
    st.write('WIP - Server Connection')
    # enter server ip - load default value from config
    # get server health
    # connect to server
    # server needs to handle multiple 'rooms'
    # create or join room
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
    if st.session_state['uploaded_data'] is None:
        st.info("Please upload a file before continuing")
        return
    
    if st.session_state['result_pvals'] is None:
        button = st.button("Process Data!")
    else:
        button = st.button("Reprocess Data!")
    
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

    if st.session_state['result_pvals'] is not None:
        df_pvals = st.session_state['result_pvals']
        labels = st.session_state['result_labels']
        with st.expander('Variable Mapping'):
            for id2label in [f'{i+1} $\\rightarrow$ {l}' for i,l in enumerate(labels)]:
                st.markdown("- " + id2label)
        st.dataframe(dataframe_explorer(df_pvals), use_container_width=True)
    return

def step_transfer_data():
    if st.button('Send data to server!'):
        with st.spinner('Data is being sent to server...'):
            # Post pval data to server
            st.write('eyo')
        with st.spinner('Waiting for other participants to provide their data...'):
            # repeatedly query server for missing data uploads, allow cancellation 
            st.write('eyo')
        with st.spinner('Data is being processed...'):
            # wait for server response
            st.write('eyo')
    # store result in session state
    return

def step_view_results():
    # use visualization library to show images of pags as well
    return


def main():
    st.write('# Welcome to {Some App}')
    
    step = stx.stepper_bar(steps=["Connect to Server", "Upload Data", "Process Data", "Transfer Data", "View Result"])
    
    if st.session_state['current_room'] is not None:
        show_room_sidebar()
    
    if st.session_state['stepper_state'] == None:
        st.session_state['stepper_state'] = step
    else:
        step = st.session_state['stepper_state']

    if step == 0:
        step_connect_to_server()
    elif step == 1:
        step_upload_data()
    elif step == 2:
        step_process_data()
    elif step == 3:
        step_transfer_data()
    elif step == 4:
        step_view_results()
    else:
        st.error('Please reload the page - Stepper out of range')

        
main()