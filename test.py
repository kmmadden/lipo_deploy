import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict
import plotly.graph_objects as go
import plotly.express as px

torch.autograd.set_grad_enabled(False)

st.set_page_config(layout="wide")
st.title('Lipohypertrophy Prediction')

#load the model
@st.cache(suppress_st_warning=True)
def create_model():
    cnn_model = models.densenet121(pretrained=True)
    new_layers = nn.Sequential(OrderedDict([
                ('new1', nn.Linear(1024, 500)),
                ('relu', nn.ReLU()),
                ('new2', nn.Linear(500, 1))
            ]))
    cnn_model.classifier = new_layers

    cnn_model.load_state_dict(torch.load('densenet_final.pth', map_location=torch.device('cpu'))) #put the directory here where cnn_model.pt is located
    return cnn_model

@st.cache(suppress_st_warning=True)
def create_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    return model