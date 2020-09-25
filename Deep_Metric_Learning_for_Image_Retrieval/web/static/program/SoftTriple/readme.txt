Put code below in Hello.py:

import SofttripleRetrieval # This python file is in the same directory as Hello.py
model = SofttripleRetrieval.retrieve(
            sys.path[0] + '/' + "static/program/SoftTriple/softtriple_cub_sample_embedding_space.csv",
            sys.path[0] + '/' + "static/program/SoftTriple/softtriple_cub_sample_image_path.csv",
            sys.path[0] + '/' + "static/program/SoftTriple/softtriple_cub_sample_state.pth")
result = model.get_10_image_path(new_image_path)
print(result)