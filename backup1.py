        
        def _step(input, row, row_state, row_cell, col_state, col_cell): #[row_states] + [col_state]
            #'''it needs the left, and upper grid as it's predecessor,  
            #   it only update the col_state after process one local patch.
            #   although it needs two states to compute each step, but it actually only updates the upper column state, for the
            #   left row_state, it only use it.
            #'''
            states = [col_state]
            cells = [col_cell]
            output, new_states, new_cells = step_function(input, row_state, states[0], row_cell, cells[0])  
            # this function actually should only return one state

            if masking:
                # if all-zero input timestep, return
                # all-zero output and unchanged states
                switch = T.any(input, axis=-1, keepdims=True)
                output = T.switch(switch, output, 0. * output)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(switch, new_state, state))
                return [output] + [return_states] + [new_cells]
            else:
                return [output] + [new_states] + [new_cells]

        def loop_over_col(coldata, col_ind, col_state, col_cell, rows_states, rows_cells, rows_model):
            #'''when loops over a column, it needs the left column as row_states,
            #row_states, col_states are already concatenated to be tensors for scan to process before call this function.
            #it return the results as well as this column response as the row_states to next column.
            #it should return the last column_state
            #'''
            # The received coldata is row, nsample, dim
            # This scan will scan over the row, and each scanned iterm will be nsample, dim
            results , _ = theano.scan( fn = _step,
                                       sequences = [coldata, rows_model, rows_states, rows_cells ],                                     
                                       outputs_info=[None] +  [col_state] + [col_cell]
                                     )
            new_row_states = [results[1]] #'''list of tensor of row_size * nsample * out_dim, but we need to modify it to be list'''
            new_row_cells = [results[2]]   
            col_vals = results[0]  
           #'''tensor of row_size * nsample * out_dim'''
           #'''the length is the number of states for each single actual step'''
            single_state_num = len(new_row_states) 
            returned_row_states = []
            returned_row_cells = []
            for timestep_id in range(self.grid_shape[0]):       
                #'''new_row_states is a list of tensor of size row_size * nsample * out_dim
                #   returned states should be [timestep_1_state_1,...,timestep_1_state_n, ...., timestep_m_state_1,...,timestep_m_state_n],
                #   n is the number of required states for each single actual step function, most time it is 1. but CWRNN has 2.
                #   m is the row_size, = grid_shape[0]
                #'''
               for state_id in range(single_state_num):
                    returned_row_states.append(T.squeeze(new_row_states[state_id][timestep_id]))
                    returned_row_cells.append(T.squeeze(new_row_cells[state_id][timestep_id]))
            
            returned_col_state = []
            for state_id in range(single_state_num):
              returned_col_state.append(T.squeeze(new_row_states[state_id][-1]))
            #'''output should be list corresponding to the outputs_info'''
            return [col_vals] + [T.stack(returned_row_states, axis = 0)] + [T.stack(returned_row_cells, axis = 0)] + [T.stack(returned_col_state, axis = 0)]    
            
