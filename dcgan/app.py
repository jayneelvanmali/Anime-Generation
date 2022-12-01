import torch
from flask import Flask, render_template, request
import torch_utils1

app = Flask(__name__)
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    fixed_latent = torch.randn(64, 128, 1, 1)
    count = 0
    torch_utils1.save_samples(count, fixed_latent)
    return render_template('home.html') 

@app.route('/', methods=['GET','POST'])
def home():
    
    
    return render_template('home.html')
    
  


    

if __name__ == "__main__":
    app.run()    