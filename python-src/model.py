from mpi4py import MPI

class DistConv:
    def __init__(self, in_dim: int, out_dim: int, with_bias: bool = False) -> None:
        # TODO: init
        pass
    
    def forward(self):
        pass

class DistModel:
    def __init__(self, config: dict, in_dim: int) -> None:
        self.input_dim = in_dim
        self.output_dim = config["model_properties"]["classes"]
        self.dropout = config["model_properties"]["dropout_rate"]
        self.lay_dim = config["model_properties"]["hidden_dims"]
        self.number_of_hid_layers = len(self.lay_dim)
        self.with_bias = config["model_properties"]["with_bias"]
        
        self.layers = []
        
        if self.number_of_hid_layers > 0:
            in_conv = DistConv(self.input_dim, self.lay_dim[0], self.with_bias)
            self.layers.append(in_conv)
            for i in range(1, self.number_of_hid_layers):
                self.layers.append(DistConv(self.lay_dim[i-1], self.lay_dim[i], self.with_bias))
            out_conv = DistConv(self.lay_dim[-1], self.output_dim, self.with_bias)
            self.layers.append(out_conv)
        else:
            out_conv = DistConv(self.input_dim, self.output_dim, self.with_bias)
            self.layers.append(out_conv)
       
       
    def forward(self):
        pass 