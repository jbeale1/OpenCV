# OpenCV
Experiments with OpenCV for detecting objects

BlobTrack1.py does simple motion tracking (blob velocity and distance-travelled). Sample video from  at https://youtu.be/KCevhaR75_s
It is so simple that it does not actually match blobs from one frame to the next, but it still works when only one moving object is in frame.  It would be better to compare position, size and speed to keep each blob matching the same object.
