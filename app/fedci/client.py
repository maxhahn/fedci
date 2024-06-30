

class Server:
    clients = []

    def start_ci_test(self):
        pass
    
    def _init_beta0(self):
        pass
    
    

class Client:
    def __init__(self, server_url, server_port, server_api_path):
        self.server_url = server_url
        self.server_port = server_port
        self.server_api_path = server_api_path
        
    def _connect(self):
        pass
    
    def start_ci_test(self):
        pass
    
    def _init_eta(self, x, beta0):
        pass
    
    def _init_mu(self, eta):
        pass
    
    def _init_deviance(self, y, mu, w):
        pass
    
    def _step(self, ):
    
    
     
        
        