class Config: 
    
    def __init__(self):
       
        self.sample_rate = 16000
        self.n_fft = 1024 
        self.win_length = int(self.sample_rate * 0.025) 
        self.hop_length = int(self.sample_rate * 0.010) 
        self.n_mels = 128
        self.f_max = 8000
        self.f_min = 0
        
        # --- FIXED PATHS FOR YOUR NEW CLEAN WORKSPACE ---
        self.data_info_dict = {
            'train': {
                'wav_dir': '/home/siplabiith',
                'alignments_path': '/home/siplabiith/clap_dwd_overall/data/train/train_overall_combined.pkl'
            },
            'test': {
                'wav_dir': '/home/siplabiith',
                # Points to test.pkl, which we will swap between tamil_iv.pkl and tamil_oov.pkl during evaluation steps
                'alignments_path': '/home/siplabiith/clap_dwd_overall/data/test/test_overall_iv_combined.pkl'
            }
        } 
        
        self.batch_size = 128
        self.hidden_dim = 256
        self.embedding_dim = 512
        self.number_examples_wer_word = 4
        self.no_of_tokens = 73
        self.input_dim_text = 256
        self.num_epochs = 30
        self.wt_clap_loss = 1.0
        self.wt_dwd_loss = 0.5
        self.num_layers = 3