# Body-Pose-Tracker

Welcome! This repository is dedicated to track human body posture leveraging deep learning techniques, specifically utilizing CNN. In our notebook script, main.ipynb,' delve into code implementing regression to identify 14 key points of the human body. Our findings and insights are documented in the reports.

Inside our 'src' folder, you'll discover functions designed for use in the 'main.ipynb' notebook. These functions, including 'show_keypoints' for visualizing keypoints and 'create_custom_dataset,' align with their suggestive names. Additionally, 'rescale,' 'crop,' and 'to_tensor' are implemented to execute data transformations before inputting into the network. Furthermore, 'convpose' encompasses the model architecture.

After approximately 15 epochs, our model achieved an impressive MSE of 0.0878. It's important to mention that, owing to size limitations on GitHub, we regretfully cannot share the model weights. Currently, post regression point identification, our focus is on connecting these points to visualize the body posture.

Stay tuned for updates as we continue to innovate and enhance our methodologies in this exciting domain!


