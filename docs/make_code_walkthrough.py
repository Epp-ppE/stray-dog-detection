"""Build docs/code_walkthrough.pdf: every source file, shown in small groups of lines,
each group with a plain-English explanation. Run from the repo root:

    python docs/make_code_walkthrough.py      (needs Playwright + Chromium for the PDF)

Line ranges below refer to the files as committed; the script fails loudly if a file
has changed length, so the notes can't silently drift out of sync with the code.
"""
import html, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (file, one-paragraph purpose, expected line count, [(first_line, last_line, explanation), ...])
FILES = [
("src/config.py", "Settings shared by every other file: where folders are and a few numbers used everywhere. Changing a value here changes it for the whole pipeline.", 13, [
 (1, 1, "A docstring: a description of the file for people. Python ignores it when running."),
 (2, 2, "Imports <code>Path</code>, a tool for building file paths that works the same on Windows, Mac and Linux."),
 (4, 4, "<code>ROOT</code> is the repository folder. <code>__file__</code> is this file's own location (<code>src/config.py</code>); <code>.resolve()</code> makes it a full path; <code>.parents[1]</code> goes up two levels, from the file to <code>src/</code> to the repository."),
 (5, 9, "Folder and file locations, built with <code>/</code>, which joins path pieces. <code>RAW</code>: the unzipped footage. <code>OUT</code>: small results that are committed to GitHub. <code>WORK</code>: big temporary files that are not committed. <code>MODELS</code>: model weight files. <code>MANIFEST</code>: the table with one row per video."),
 (11, 11, "Thailand's time zone is UTC+7. Used when turning filename timestamps into local time."),
 (12, 12, "How many frames per second of video are analysed. 3 is a balance: enough to follow a walking dog, about 5x less work than all 15."),
 (13, 13, "Frames are shrunk to 1280 pixels wide (1280x720) before detection. The originals are 2560x1440."),
]),
("src/video.py", "The frame sampler. Every step that needs video frames (detection, choosing evaluation frames) uses this one function, so frame number <i>i</i> always means exactly the same picture everywhere.", 26, [
 (1, 5, "Docstring: decoding H.265 video at 2560x1440 is slow, so the program ffmpeg drops unwanted frames first and only shrinks the ones that are kept."),
 (6, 7, "<code>subprocess</code> runs another program (ffmpeg) from Python. <code>numpy</code> handles arrays of numbers, which is how images are stored."),
 (10, 10, "Defines the function <code>iter_frames</code>. It is a <i>generator</i>: instead of returning all frames at once (which would fill the memory), it hands them back one by one as the caller asks for them."),
 (11, 13, "Builds the ffmpeg command. <code>-v quiet</code>: no log messages. <code>-i</code>: the input video. <code>-an</code>: ignore the audio. <code>-vf</code>: video filters, first <code>fps=3</code> (keep 3 frames per second), then <code>scale</code> to 1280x720 using the 'area' method, which gives clean downscaling. <code>rawvideo</code> + <code>bgr24</code>: output plain pixel values, 3 bytes per pixel in Blue-Green-Red order (the order OpenCV expects). <code>-</code>: send the output to Python instead of a file."),
 (14, 14, "Starts ffmpeg and connects its output to a pipe Python can read from, with a large buffer for speed."),
 (15, 16, "<code>n</code> = the number of bytes in one frame (width x height x 3 colours). <code>i</code> counts frames from 0."),
 (17, 23, "Loop: read exactly one frame's worth of bytes. If fewer bytes come back, the video has ended, so stop. Otherwise <code>yield</code> three things: the frame number, its time in seconds (<code>i / fps</code>, so frame 6 at 3 fps is 2.0 s) and the picture, reshaped from a flat list of bytes into a height x width x 3 image. Then move to the next frame number."),
 (24, 26, "<code>finally</code> always runs, even if the caller stops early: close the pipe and wait for ffmpeg to exit, so no stray processes are left behind."),
]),
("scripts/download_data.sh", "A shell script (runs in a terminal, not Python) that downloads the three zip files from the GitHub release <code>data-v1</code> and unzips them into <code>data/raw/</code>.", 17, [
 (1, 3, "The first line says 'run this with bash'. Lines starting with <code>#</code> are comments."),
 (4, 4, "Safety settings: stop at the first error (<code>-e</code>), treat a misspelt variable as an error (<code>-u</code>), and fail if any step of a chained command fails (<code>pipefail</code>)."),
 (5, 7, "Settings: which repository (can be overridden from outside), the release name, and the list of zip files."),
 (8, 9, "Creates the folders if missing and goes into the zips folder."),
 (10, 14, "For each zip: if it is already downloaded and not empty (<code>-s</code>), skip it. Otherwise try the GitHub command-line tool <code>gh</code>; if that is not available or fails, download with <code>curl</code>: <code>-f</code> fail on server errors, <code>-L</code> follow redirects, <code>--retry 3</code> try again up to 3 times."),
 (15, 16, "Back to the repository folder, then unzip each file into <code>data/raw</code>. <code>-q</code>: quiet. <code>-n</code>: never overwrite files that already exist."),
 (17, 17, "Prints how many files ended up in <code>data/raw</code>."),
]),
("src/clock_ocr.py", "Reads the date and time burned into the top-left of every frame. This was needed because Camera1's filenames are export times, not recording times. General text recognition (Tesseract) failed on bright backgrounds, so this file recognises each digit separately by its shape.", 106, [
 (1, 13, "Docstring explaining the reason for this approach, as above."),
 (14, 18, "Imports: <code>datetime</code> for dates and times, <code>subprocess</code> for ffmpeg, <code>numpy</code> for arrays, <code>cv2</code> (OpenCV) for image processing, and <code>hog</code> from scikit-image, a standard way to describe the shape of a small image."),
 (20, 20, "The original video size, 2560x1440."),
 (21, 21, "The rectangle (left, top, right, bottom) that contains the clock text, in original-size pixels."),
 (22, 23, "Measured fact: the on-screen clock runs 0.9 seconds ahead of the time in Camera3/4 filenames."),
 (24, 24, "The clock's fixed layout: characters are 42 pixels apart (<code>PITCH</code>), the first starts at x = 128, each digit is cut out 40 pixels wide, and the text spans y = 48 to 110."),
 (25, 25, "Which of the 19 character positions in <code>2025/01/07 06:22:29</code> are digits: everything except the two slashes (positions 4 and 7), the space (10) and the two colons (13 and 16). That leaves 14 digits: YYYY MM DD HH MM SS."),
 (28, 29, "Function that decodes only the clock area, from every n-th frame of a clip."),
 (30, 31, "Unpacks the rectangle and works out its width and height."),
 (32, 34, "ffmpeg command. <code>select=...</code> keeps only frames whose number divided by <code>every_n</code> leaves remainder <code>phase</code> (with 15 and 7: frames 7, 22, 37, ...). <code>crop</code> cuts out the clock strip, <code>format=gray</code> makes it black and white, and <code>passthrough</code> keeps the original frame timing."),
 (35, 36, "Runs ffmpeg, collects the raw bytes, and reshapes them into a stack of clock-strip images."),
 (37, 38, "Works out each strip's time in seconds: (strip number x every_n + phase) / 15 fps. Returns the times and the strips."),
 (41, 42, "Function that cuts the 14 digits out of one clock strip and describes each one's shape."),
 (43, 46, "For each digit position: its left edge inside the strip (<code>xa</code>), then slices that cell out, from y = 48 to 110."),
 (47, 48, "Resizes every cell to the same 32x48 size, then <i>equalises</i> its brightness: stretches the contrast so a digit on bright sky and a digit at night end up looking alike."),
 (49, 50, "HOG (Histogram of Oriented Gradients) describes the cell by the direction of its edges in small 8x8 patches, which captures the digit's shape and ignores colour and brightness. The 14 descriptions are stacked and returned."),
 (53, 54, "Turns a date-time into its 14 digits. For example, 2025-01-07 06:22:29 becomes [2,0,2,5,0,1,0,7,0,6,2,2,2,9]. Used to make training labels."),
 (57, 60, "The digit reader, a <i>k-nearest-neighbours</i> classifier: to recognise a digit, find the k = 3 most similar digits it has seen before and let them vote. <code>X</code> stores the example shapes and <code>y</code> their correct digits."),
 (62, 66, "<code>fit</code> ('training'): for k-NN this just stores the examples. Each is scaled to length 1 so that comparing two of them measures shape similarity only."),
 (68, 74, "<code>predict_digits</code>: scale the new cells the same way, compute their similarity to every stored example in one matrix multiplication (<code>@</code>), take the 3 most similar (<code>argsort</code>), and return the digit that wins the vote (<code>bincount(...).argmax()</code>)."),
 (76, 81, "<code>read</code>: predicts the 14 digits, joins them into text like <code>20250107062229</code> and turns that into a date-time. If the result is impossible (for example month 13), it returns <code>None</code> instead of crashing."),
 (83, 92, "<code>clip_start</code>: reads the clock in every 15th frame of a clip (frames 7, 22, 37, ..., i.e. half-way through each second). For each successful reading it subtracts that frame's time to get the implied clip start. If nothing could be read, it returns nothing."),
 (93, 96, "Takes the <i>median</i> of all implied start times, so an occasional misread frame doesn't matter. Adds 0.5 s, because a clock showing whole seconds is on average half a second behind, and subtracts the 0.9 s lead. Also counts how many frames agree within 2 seconds, as a confidence check. Returns the start time, the agreeing count and the frame count."),
 (98, 106, "<code>save</code> / <code>load</code>: store the trained reader in a compressed <code>.npz</code> file and read it back."),
]),
("src/train_clock_reader.py", "Trains the digit reader from <code>clock_ocr.py</code> and measures how accurate it is. The training labels are free: Camera3/4 filenames already give the true time of every frame.", 57, [
 (1, 8, "Docstring: how the labels are made and how accuracy is checked (30% of clips held back)."),
 (9, 13, "Imports. <code>ThreadPoolExecutor</code> runs 4 jobs at the same time to use all CPU cores. <code>import clock_ocr as c</code> gives that file the short name <code>c</code>."),
 (15, 15, "A time-zone object for UTC+7."),
 (18, 23, "<code>load_clip</code>: the first 13 characters of a Camera3/4 filename are milliseconds since 1 January 1970, which gives the clip start; convert it to Thai local time. Then read clock strips at phase 7, describe the digits (<code>X</code>) and make the correct answers (<code>Y</code>): the digits of (start + frame time + 0.9 s lead), rounded down to whole seconds."),
 (26, 30, "Collects Camera3 and Camera4 videos whose name is exactly 13 digits. That skips the <code>(1)</code>/<code>(2)</code> duplicates and the re-shared file."),
 (31, 32, "Loads all those clips, 4 at a time."),
 (33, 35, "Fixed random seed, so the split is the same every run. Shuffles the clip order and sets aside the first 30% of clips as the <i>test</i> set. The split is by clip, so the test frames come from videos the reader never saw."),
 (36, 37, "A small helper that merges all digit cells and labels from a list of clips into one big table."),
 (38, 38, "Trains a reader on the other 70% of clips."),
 (39, 45, "Tests it: for every test frame, the share of digits that are right and whether the whole timestamp is right; for every test clip, how far its computed start time is from the true filename time."),
 (46, 51, "Puts the results in a dictionary, prints them and saves them to <code>outputs/clock_reader_eval.json</code>. Result: 99.5% of digits right, every clip start within 0.13 s."),
 (52, 53, "Trains the final reader on <i>all</i> clips (more examples is better) and saves it to <code>models/clock_knn.npz</code>."),
 (56, 57, "Standard Python ending: run <code>main()</code> only when this file is run directly, not when another file imports it."),
]),
("src/prepare.py", "Step 1 of the pipeline. Builds <code>outputs/manifest.csv</code>, a table with one row per video file: its size and length, whether it is a duplicate, and its real recording start time.", 85, [
 (1, 8, "Docstring summarising the three jobs of this file."),
 (9, 14, "Imports: <code>hashlib</code> for file fingerprints, <code>json</code>/<code>csv</code> for reading and writing data, <code>re</code> for text patterns, plus the clock reader and settings."),
 (17, 23, "<code>probe</code>: runs <code>ffprobe</code> (ffmpeg's inspection tool), which reports a video's details as JSON. Finds the video stream and converts the frame rate from a fraction like <code>15/1</code> to a number. Returns duration, fps, width, height and codec."),
 (26, 31, "<code>md5</code>: reads the file in 1 MB pieces and feeds them into an MD5 fingerprint. Two files get the same fingerprint only if every byte is identical, which is how the 22 duplicates were found."),
 (34, 36, "Lists every <code>.mp4</code> under <code>data/raw/</code>. <code>.lower()</code> also catches the one file ending in capital <code>.MP4</code>."),
 (37, 38, "Inspects and fingerprints all 447 files, 4 at a time."),
 (39, 45, "Makes one row per file. The id is the camera short name plus the filename, e.g. <code>cam4_1736159931000(1)</code>. <code>first</code> remembers the first file seen with each fingerprint, so later files with the same fingerprint are marked as duplicates of it."),
 (46, 53, "Fixes which copy counts as the original: groups the rows by fingerprint and sorts each group so the cleanly named file (exactly 13 digits, no <code>(1)</code>) comes first. Every other file in the group is marked a duplicate of it."),
 (55, 55, "Loads the trained clock reader."),
 (57, 69, "<code>start_time</code> for one row. Duplicates are skipped. Camera3/4 files with a 13-digit name get their time straight from the filename. Everything else (Camera1) gets it from the clock reader, which also records how many frames agreed, e.g. <code>clock_ocr (19/20 frames agree)</code>."),
 (71, 72, "Works out start times for all rows, 4 at a time."),
 (73, 79, "Writes the manifest table to CSV with these columns, in this order."),
 (80, 81, "Prints a summary: 447 files, 425 unique, 22 duplicates."),
 (84, 85, "Runs <code>main()</code> when the file is run directly."),
]),
("src/detect.py", "Step 2. Runs the pretrained YOLO detector on 3 frames per second of every unique clip and saves every box it finds. This is the slow step: about 4.5 hours for the final YOLO11m at 960 px on this computer (47 minutes for YOLO11s at 640 px).", 82, [
 (1, 10, "Docstring. Two key choices: boxes down to 0.01 confidence are kept, so the threshold can be tuned later without re-running the model, and other animals and people are kept, because puppies are often labelled 'cat' and night-time dogs 'sheep'."),
 (11, 15, "Imports, including <code>YOLO</code> from the ultralytics library and the frame sampler."),
 (17, 17, "COCO class numbers to keep: 0 person, 15 cat, 16 dog, 17 horse, 18 sheep, 19 cow, 21 bear. Everything else (cars, chairs, ...) is ignored."),
 (18, 19, "Keep boxes with confidence of at least 0.01, because the best setup's tuned threshold is about 0.02. Process frames in batches of 8, which is faster than one at a time."),
 (22, 24, "<code>colourfulness</code>: looks at every 8th pixel and measures how different its red, green and blue values are, then averages. Infrared night frames are grey (all three equal), so they score close to 0. This is how night frames are recognised."),
 (27, 31, "<code>run</code>: makes a name like <code>yolo11m_960</code>, a folder for this model's results, and loads the model weights."),
 (32, 33, "Reads the manifest and keeps only unique clips (rows with no <code>duplicate_of</code>)."),
 (34, 38, "Puts the 115 clips that contain hand-labelled frames first, so tracker tuning can start before the whole run finishes."),
 (39, 43, "Goes through the clips. If a clip's result files already exist, it is skipped, so an interrupted run can be restarted and continues where it stopped."),
 (44, 44, "Empty lists for this clip's detections, frame information, and the current batch of frames."),
 (46, 55, "<code>flush</code>: runs YOLO on the batch. <code>imgsz</code> is the size the model works at (640 or 960), <code>conf</code> the minimum confidence and <code>classes</code> the kept classes. For every frame and every box it records the clip, frame number, time, class name, confidence and the box corners as fractions of the image size (0 to 1), then empties the batch."),
 (57, 63, "Reads the clip's frames through the sampler (3 fps, 1280x720). For each frame: record its colourfulness, add it to the batch, and run the model whenever 8 are waiting. After the last frame, run the model on any leftovers."),
 (64, 65, "Saves this clip's detections and frame list as two small CSV files."),
 (66, 68, "Prints progress every 20 clips."),
 (69, 74, "After all clips: combines the per-clip files into one compressed detections file (<code>data/work/</code>) and one frames file (<code>outputs/frames.csv.gz</code>), then prints totals."),
 (77, 82, "Command-line options, e.g. <code>python detect.py --model yolo11m --imgsz 960</code>."),
]),
("src/track.py", "Step 3. Links the boxes of one dog across frames into a <i>track</i> using ByteTrack, then keeps only tracks that look like real dogs. Takes about a minute because it works on the saved detections and doesn't run the detector again.", 113, [
 (1, 10, "Docstring: why tracks are filtered (one-frame flickers are the most common false alarm)."),
 (11, 17, "Imports. <code>BYTETracker</code> is the ByteTrack implementation inside ultralytics. <code>SimpleNamespace</code> bundles settings into one object. <code>merge_animals</code> comes from <code>boxes.py</code>."),
 (19, 19, "Frame size in pixels: 1280 x 720."),
 (20, 22, "Default tracker settings. Boxes above <code>track_high_thresh</code> are matched to tracks first; boxes between the low and high thresholds are used in a second pass only to keep existing tracks alive (ByteTrack's key idea: a dog that briefly looks unclear isn't lost). A new track needs <code>new_track_thresh</code>. A lost track is remembered for 12 frames (4 seconds), to survive a dog passing behind the fence or plants. <code>match_thresh</code> limits how badly a box may overlap a track's predicted position and still match. <code>fuse_score</code> also uses confidence when matching. For the final setup these are replaced by tuned values (see below)."),
 (23, 24, "Default confirmation rule: a track counts as a dog if it lasts at least 3 frames (1 second) and was at least 0.45 confident at some point."),
 (25, 25, "Which classes count as 'dog' when other animals are not merged."),
 (26, 27, "Where <code>tune_tracker.py</code> writes the settings it chose on the dev clips."),
 (30, 44, "<code>Dets</code>: a small adapter that gives BYTETracker data in the shape it expects. It holds boxes, confidences and classes, can report boxes as centre-x, centre-y, width, height (<code>xywh</code>), can say how many boxes it has, and can return a subset chosen by a True/False list."),
 (47, 50, "<code>track_video</code>: a fresh tracker for each clip, so track numbers restart in every video. Groups the clip's detections by frame number."),
 (51, 57, "Goes through <i>every</i> frame number, including frames with no detections, because the tracker must see empty frames to count how long a dog has been missing. Converts boxes from fractions back to pixels."),
 (58, 60, "Updates the tracker with this frame's boxes. It returns the tracks visible now, each with a box, a track id and a score, stored as fractions again."),
 (63, 69, "<code>load_detections</code>: reads all saved boxes. With <code>animals</code>, every animal class counts as dog and duplicate boxes on the same animal are removed; otherwise only 'dog' boxes are kept."),
 (72, 73, "<code>run_tracking</code>: tracks every clip and marks which tracks are confirmed. Also used by <code>tune_tracker.py</code>."),
 (74, 80, "Tracks each clip separately (clips with no boxes get an empty table) and labels the results with the clip id."),
 (81, 83, "Combines all clips into one table, adds each row's time in seconds, and gives each track a unique name such as <code>cam3_1736153024000:14</code>."),
 (84, 87, "For each track: how many frames long it is and its best score. Tracks passing both rules are marked <code>confirmed</code>."),
 (90, 93, "<code>main</code>: loads the boxes and finds how many frames each clip has."),
 (94, 100, "Starts from the defaults, then, if the tuned settings were made for this same detector setup, uses those instead and says so."),
 (101, 105, "Tracks everything, saves to <code>outputs/tracks_&lt;setup&gt;.csv.gz</code> and prints how many tracks were confirmed in how many clips."),
 (108, 113, "Command-line options: <code>--det</code> chooses the detector run, <code>--merge-animals</code> counts all animal classes as dog."),
]),
("src/select_eval_frames.py", "Picks the 180 frames you labelled, at random and <i>before</i> looking at any model output, so the test can't be tilted towards easy frames. Then saves those frames as images.", 78, [
 (1, 10, "Docstring: how frames are chosen and why the images match the detector's input exactly."),
 (11, 14, "Imports."),
 (16, 16, "A fixed random seed: running the script again picks exactly the same 180 frames."),
 (17, 20, "The plan: for each camera and day/night group, how many frames to pick and the most allowed from a single clip. Camera1: 50 day + 20 night. Camera3: 45 day + 15 night. Camera4: 50 day (it has no night clips)."),
 (21, 21, "Frames from the same clip must be at least 3 seconds apart, because neighbouring frames are near-copies."),
 (24, 29, "<code>clip_is_night</code>: decodes a small 320x180 version of the clip's first sampled frame and checks whether it is grey (colour spread under 4), i.e. infrared."),
 (32, 36, "Loads the unique clips and marks each as day or night. This decodes every clip, so it is the slow part."),
 (37, 41, "For each group: lists its clips, shuffles them, and puts the first half in <i>dev</i> and the rest in <i>test</i>. Every frame of a clip therefore lands on the same side."),
 (42, 53, "The random picking loop: choose a random clip; skip it if it has reached its limit; choose a random frame within it; skip it if it is within 3 seconds of a frame already chosen from that clip; otherwise keep it. Repeat until the group has enough frames (with a safety stop after 20,000 tries)."),
 (54, 57, "Makes one row per chosen frame: an id such as <code>cam3_1736153024000__0039</code> (clip plus frame number), its time, camera, day/night and dev/test."),
 (58, 58, "Shuffles the row order, so the labelling page mixes cameras instead of showing one camera at a time."),
 (59, 68, "Saves the chosen frames as JPEG images. Each clip is decoded with the same sampler as <code>detect.py</code>, so the image you labelled is exactly what the model saw. Decoding stops after the last needed frame."),
 (69, 72, "Drops any frame that failed to save (none did) and writes the list to <code>outputs/eval/eval_frames.csv</code>."),
 (73, 74, "Prints counts per group."),
 (77, 78, "Runs <code>main()</code> when the file is run directly."),
]),
("src/stats.py", "Step 4. Turns the confirmed dog tracks into behaviour statistics per camera: visits, dogs per visit, time on screen, moving versus resting, closeness between dogs, plus two charts.", 147, [
 (1, 18, "Docstring defining every statistic. Read it once; the rest of the file implements these definitions."),
 (19, 24, "Imports. <code>matplotlib.use(\"Agg\")</code> makes charts draw straight to image files without needing a screen."),
 (26, 28, "Settings: clips less than 120 s apart belong to the same visit; a dog moving more than 0.25 body lengths per second counts as moving; short camera names for the charts."),
 (31, 35, "Loads the manifest (unique clips only) and works out each clip's start and end time."),
 (36, 37, "Loads the frame list and marks a frame as night if its colourfulness is under 4."),
 (38, 41, "Loads the tracks, keeps confirmed ones, and computes each box's centre (<code>cx</code>, <code>cy</code>), width and height."),
 (43, 44, "Counts how many confirmed dogs are in each frame."),
 (45, 49, "Builds a table with one row per clip: night if over half its frames are infrared; <code>dogs</code> = the most dogs in any single frame (a safe lower bound on distinct dogs); <code>dog_seconds</code> = sum of dogs per frame / 3 (each frame stands for a third of a second); number of tracks. <code>fillna(0)</code> gives clips with no dogs a 0 instead of a blank."),
 (51, 53, "Sorts track rows by track and time, then groups by track."),
 (54, 54, "How far each box centre moved since the previous frame, divided by the seconds in between. The vertical part is multiplied by 9/16 because positions are fractions of width and height, and the frame is 16:9."),
 (55, 56, "Divides that speed by the dog's box height, giving body lengths per second, so a big nearby dog and a small far dog can be compared. Above 0.25 counts as moving."),
 (57, 58, "Per clip, the share of dog time spent moving."),
 (60, 73, "Interactions: in every frame with at least 2 dogs, checks every pair. If their centres are closer than the average of their widths (touching, face to face, eating together), the frame counts as an interaction. Totals the seconds per clip."),
 (75, 76, "Saves the per-clip table to <code>outputs/stats/clips.csv</code>."),
 (78, 95, "Visits: per camera, goes through the clips with dogs in time order. If a clip starts within 120 s of the current visit's end, it extends that visit (later end, one more clip, largest dog count, more dog-seconds). Otherwise the current visit is closed and a new one starts. Adds each visit's length and starting hour, and saves to <code>visits.csv</code>."),
 (97, 114, "Per-camera summary: clips, minutes of footage, share of clips with dogs, recording span, visits and visits per day, typical visit length, most dogs at once, dog-minutes, share of time moving, interaction minutes and night visits. Saves to <code>summary.json</code> and prints it as a table."),
 (116, 130, "Chart 1, activity by hour: one panel per camera, with grey bars for minutes of footage recorded in each hour of the day and green bars for dog-minutes. Saved as <code>activity_by_hour.png</code>."),
 (132, 141, "Chart 2, timeline: one row per camera, a small grey dot for every clip and a green dot for every visit, bigger when more dogs were present. Saved as <code>visit_timeline.png</code>."),
 (144, 147, "Command-line option to choose which tracks file to use."),
]),
("src/evaluate.py", "Step 5. Compares each detector setup against your hand labels and reports precision, recall, F1 and AP@0.5 with 95% intervals, plus breakdowns by camera and day/night.", 205, [
 (1, 21, "Docstring: the full evaluation protocol. These are the rules quoted in the report."),
 (22, 25, "Imports. The box overlap function and the 'every animal = dog' merge come from <code>boxes.py</code>."),
 (27, 29, "The 5 model setups to compare; the evaluation folder; a fixed random generator, so the bootstrap intervals are the same every run."),
 (32, 37, "<code>predict</code>: if this setup's predictions were already saved, load them instead of re-running the model."),
 (38, 40, "Splits a name like <code>yolo11m_960</code> into model and input size, loads the model and its class names."),
 (41, 48, "Runs the model on each of the 180 evaluation images with a very low threshold (0.01), so every possible box is kept and any threshold can be tested later. Keeps animals and people."),
 (49, 51, "Saves the predictions to <code>outputs/eval/preds_&lt;setup&gt;.csv</code> and returns them."),
 (54, 60, "<code>tracked_preds</code>: the boxes the <i>whole pipeline</i> keeps after tracking (confirmed tracks only) on the evaluation frames, so tracking can be scored too. The tracks come from a saved file, or are passed in directly by the tracker-tuning script. The track's score is used as its confidence."),
 (63, 72, "<code>match</code>, the heart of scoring. For each frame: splits your boxes into normal ones (<code>G</code>) and Hard ones (<code>Hd</code>), counts the normal ones, and prepares a list marking which have been matched."),
 (73, 75, "If the model drew no boxes in this frame, there is nothing to match, and every dog in it becomes a miss."),
 (76, 84, "Goes through the model's boxes from most to least confident. Overlap (IoU) of 0.5 or more with an unmatched normal box: a true positive, and that box is now used, so the model can't get credit twice for one dog. Otherwise, overlap with a Hard box: ignored. Otherwise: a false positive (false alarm)."),
 (85, 85, "Returns every scored box's confidence, whether it was correct, and the number of real dogs."),
 (88, 100, "<code>ap_and_curve</code>: sorts boxes by confidence and walks down the list. At each step, recall = dogs found so far / all dogs and precision = correct boxes so far / boxes so far. The precision line is smoothed so it never goes up again (the standard rule), and the area under the precision-recall curve is the AP."),
 (103, 109, "<code>at_threshold</code>: keeps boxes at or above one confidence threshold and counts true positives, false positives and misses (false negatives), then precision, recall and F1 (their balance)."),
 (112, 115, "<code>best_threshold</code>: tries every distinct confidence value as a threshold and picks the one with the best F1. If two tie, it picks the lower one."),
 (118, 133, "<code>bootstrap</code>: 1,000 times, draws a new set of test frames at random <i>with replacement</i> (some frames twice, some not at all) and rescores. The middle 95% of those 1,000 scores is the reported interval, which shows how much the result could change with a different sample of frames."),
 (136, 144, "<code>main</code>: loads your labels, keeps frames marked done (skipped frames drop out), splits them into dev and test, and counts normal and Hard boxes."),
 (146, 156, "Builds the setups to compare: each model's 'dog' boxes; three models with every animal class counted as dog; and the full pipeline after tracking (baseline and final), if their track files exist."),
 (158, 172, "For each setup: picks the threshold on <b>dev</b> and records dev AP and F1 (used to choose the final setup), then scores <b>test</b> with that threshold: AP, precision, recall, F1, intervals, and results at the common default threshold of 0.25."),
 (173, 182, "Breakdowns by camera and by day/night, using all labelled frames for steadier numbers."),
 (183, 184, "Prints one line per setup and saves everything to <code>outputs/eval/metrics.json</code>."),
 (186, 193, "Draws all precision-recall curves on one chart: <code>outputs/eval/pr_curves.png</code>."),
 (196, 205, "Command-line options: <code>--boot</code> sets the number of bootstrap rounds; <code>--predict-only</code> just runs the models, without scoring."),
]),
("src/boxes.py", "Small box tools shared by tracking, evaluation and the gallery. This is where 'every animal class counts as dog' is implemented.", 33, [
 (1, 2, "Docstring and numpy import. A box is four numbers: left, top, right, bottom."),
 (4, 4, "The COCO classes treated as 'dog'. Puppies are often labelled cat, and dogs at night sheep, cow, horse or bear."),
 (7, 16, "<code>iou</code>: Intersection over Union of box <code>a</code> with every box in <code>B</code>: the overlapping area (0 if they don't touch) divided by the area the two boxes cover together. 1 = identical, 0 = no overlap."),
 (19, 22, "Docstring of <code>merge_animals</code>: after relabelling every animal as dog, one animal can have two boxes (say 'dog' and 'cat'), so the less confident one is removed."),
 (23, 23, "Keeps animal boxes only, renumbers the rows (otherwise rows from different clips can share a number and get mixed up, a bug found during tracker tuning) and sorts by confidence, highest first."),
 (24, 33, "For each frame: walk down the boxes from most confident; keep a box, then drop every remaining box overlapping it by IoU of 0.6 or more (non-maximum suppression, ignoring class). Returns the kept boxes, all labelled 'dog'."),
]),
("src/tune_tracker.py", "Chooses ByteTrack's thresholds and the 'confirmed dog' rule for the final detector, using only the dev clips. The test frames are never used.", 56, [
 (1, 10, "Docstring: why the tracker needed re-tuning. The final detector's correct boxes often score below 0.1, but the original tracker settings ignored anything under 0.35."),
 (11, 16, "Imports, including the tracking and evaluation files themselves, so exactly the same code is used."),
 (19, 24, "Loads the evaluation frames and your labels, and keeps the dev frames only."),
 (25, 30, "Reads the saved detections of the dev clips, counts every animal class as dog, and finds how many frames each clip has."),
 (31, 31, "The settings to try: 4 high thresholds x 2 low thresholds x 2 minimum lengths x 4 minimum peak scores = 64 combinations."),
 (32, 40, "For each combination: track all dev clips, take the confirmed boxes on the dev labelled frames, and score them (precision, recall, F1). The threshold is 0 because confirmation itself is the decision."),
 (41, 42, "Sorts by F1 (then precision) and saves the full table to <code>outputs/eval/tracker_grid.csv</code>."),
 (43, 48, "Saves the winner to <code>outputs/eval/tracker_params.json</code>, which <code>track.py</code> then uses."),
 (49, 50, "Prints the top 8 combinations and the choice."),
 (53, 56, "Command-line option for which detector run to tune on."),
]),
("src/examples.py", "Step 6. Draws your labels and the model's boxes on all 180 evaluation frames and sorts them into correct / false alarm / missed, for the report's success and failure examples.", 88, [
 (1, 9, "Docstring, including the colour code used in every gallery image."),
 (10, 13, "Imports, including <code>evaluate.py</code> so the matching rules are identical to the scoring."),
 (15, 16, "The colours (OpenCV uses Blue-Green-Red order) and the frame size."),
 (19, 25, "<code>preds_for</code>: gets the boxes for a setup name: after tracking, with every animal merged, or the plain 'dog' class."),
 (28, 40, "<code>box</code>: draws one box (solid, or dashed for Hard labels) with an optional label on a filled background."),
 (43, 49, "Loads the setup's dev-chosen threshold, your labels and the frame list, and keeps only boxes at or above the threshold."),
 (50, 53, "Creates the three output folders."),
 (54, 57, "For each frame: your normal and Hard boxes, and the model's boxes for that frame, most confident first."),
 (58, 67, "The same matching as the evaluation: IoU of 0.5 or more with an unmatched dog = correct; on a Hard box = ignored; otherwise a false alarm."),
 (68, 73, "Draws everything: your boxes green, Hard boxes dashed orange, correct detections yellow, false alarms red and missed dogs magenta."),
 (74, 74, "Category: 'missed' if any dog was missed, else 'false_alarm' if there was a false alarm, else 'correct'."),
 (75, 78, "Saves a smaller copy of the picture into its category folder and records the counts for this frame."),
 (79, 82, "Saves the per-frame table and prints totals per category and per camera."),
 (85, 88, "Command-line option for which setup to draw."),
]),
("src/reid_demo.py", "The optional dog-recognition demo: groups tracks that probably show the same dog. It uses a pretrained image model with no training, so the groups are candidates for a person to check, not confirmed identities.", 185, [
 (1, 15, "Docstring: the method and its limits."),
 (16, 21, "Imports."),
 (23, 26, "Settings: frame size; up to 4 crops per track; crops must be at least 12% of the frame height; two groups are merged when their average similarity is above 0.5."),
 (29, 42, "<code>best_crops</code>: keeps boxes that are big enough, not touching the image edge, and not overlapping another dog (a crop with two dogs would mix their looks). Then picks each track's 4 best crops by size x confidence."),
 (45, 57, "<code>extract</code>: decodes each needed clip and cuts out the chosen crops, with 8% padding around each box."),
 (60, 77, "<code>cluster</code>: starts with every track as its own group and repeatedly merges the two most similar groups (average similarity), except pairs marked 'cannot link'. Stops when no pair is similar enough."),
 (80, 84, "<code>colour_hist</code>: describes coat colour as a 12 x 6 hue-saturation histogram, which captures patches like 'white with tan spots'."),
 (87, 96, "<code>features</code>: subtracts the average of this camera's tracks from each feature type, so similarity reflects what makes a dog <i>different</i> from a typical dog here, then joins the image-model and colour features and scales them to length 1."),
 (99, 108, "<code>tile</code>: places crops side by side on square tiles for the output pictures."),
 (111, 119, "<code>main</code>: reads the manifest, works out which clips are night (infrared, skipped because colour is lost), and keeps confirmed daytime tracks from one camera."),
 (120, 134, "Gets the crops and their image-model features (YOLO11s classifier, 512 numbers per crop, averaged per track), from a cache file if this was already done."),
 (135, 135, "Builds the final combined features."),
 (136, 144, "'Cannot link' pairs: two tracks visible in the same frame must be different dogs."),
 (145, 145, "Clusters the tracks into candidate dogs, biggest groups first."),
 (147, 150, "Prepares the output folder and removes old pictures."),
 (151, 169, "Records every track's group and, for up to 8 groups that span two or more clips, saves a picture: one row per track, labelled with its date and time."),
 (170, 176, "Saves the group table and a summary with the counts and the caveat."),
 (179, 185, "Command-line options: which tracks, which camera, and an optional merge threshold."),
]),
("tools/import_labels.py", "Turns the labels exported from the labelling page (one file per frame) into the single <code>outputs/eval/labels.json</code> that the evaluation reads.", 23, [
 (1, 5, "Docstring and how to run it."),
 (6, 6, "Imports."),
 (8, 9, "Input folder (from the command line) and output file."),
 (10, 16, "For each exported frame: keep its status (done or skip) and its boxes, with coordinates rounded to 5 decimals and the Hard flag as true/false."),
 (17, 17, "Writes all frames to one JSON file."),
 (18, 23, "Prints a summary: frames, statuses, boxes and Hard boxes."),
]),
("scripts/run_all.sh", "Runs the whole project from scratch, in order. One command reproduces every result.", 24, [
 (1, 5, "What the script does and how to call it."),
 (6, 7, "Stop at the first error, then move to the repository folder wherever the script was called from."),
 (8, 8, "The detector to use: YOLO11m at 960 px by default, which can be changed from outside, e.g. <code>DET_MODEL=yolo11s DET_SIZE=640</code> for a quick run."),
 (10, 11, "Download the footage (unless <code>SKIP_DOWNLOAD=1</code>) and the four model weight files."),
 (12, 16, "Clock reader, manifest, evaluation images, then detection on every clip (the long step)."),
 (17, 20, "Score the detectors, tune the tracker on dev clips, track, and score again including the tracked pipeline."),
 (21, 23, "Statistics and charts, the success and failure gallery, and the recognition demo."),
 (24, 24, "Done message."),
]),
]

LABEL_TOOL = [
 ("Top of the file", "The page title, fonts from Google Fonts, and the style rules (CSS). Colours are defined once as tokens such as <code>--accent</code> and <code>--box</code>, with a second set for dark mode."),
 ("Page layout (HTML)", "The progress bar and counter; the chips showing camera, day/night and status; the 'How to label' guide; the image area (<code>#canvas</code>) holding the image, your saved boxes (<code>#boxes</code>) and the box being drawn (<code>#ghostLayer</code>); the list of boxes with delete buttons; the status line; and the toolbar buttons."),
 ("State variables", "<code>frames</code>: the 180 frames. <code>idx</code>: which one is shown. <code>labels</code>: everything saved so far. <code>cur</code>: the boxes of the current frame. <code>sel</code>: the selected box. Plus zoom, pan mode and the connection to the database."),
 ("Local drafts", "Unsaved changes are also kept in the browser's own storage (<code>localStorage</code>), so a dropped connection doesn't lose a box. The last frame you were on is remembered too."),
 ("load(i)", "Shows frame number <i>i</i>: saves the previous frame first, picks up its saved boxes (or a draft), sets the image, and updates the chips and buttons."),
 ("render()", "Redraws the boxes, labels and the box list from <code>cur</code>. Box positions are stored as fractions of the image (0 to 1), so they line up at any zoom or screen size."),
 ("touched() and flush()", "After any change, <code>touched()</code> waits 1.2 s for more changes, then <code>flush()</code> saves the frame to the database as <code>labels/&lt;frame id&gt;</code> with its boxes, status (done or skip) and the time. This is the data the evaluation reads."),
 ("Drawing", "Finger down starts a box at that point; moving stretches the blue outline; finger up creates the box. A movement under 14 px counts as a tap, which selects the smallest box under your finger. A second finger is ignored while drawing."),
 ("Buttons and keys", "Prev, Next, Next unlabelled, Hard, Delete, Zoom (1x, 2x, 3x), Draw/Pan, Skip, No dogs and Done. On a keyboard: N, P, Backspace, H, 0, Enter and Z."),
 ("Loading data", "Fetches the frame list (<code>frames.json</code>), then subscribes to the database so saved labels appear immediately and stay up to date. If the database can't be reached, a message says so."),
]

INTRO_BASICS = [
 ("<code># ...</code> and <code>\"\"\"...\"\"\"</code>", "Comments and docstrings: notes for people. Python skips them."),
 ("<code>import x</code>, <code>from x import y</code>", "Loads a library or another file of this project so its tools can be used."),
 ("<code>name = value</code>", "Stores a value under a name (a variable). ALL_CAPS names are settings that don't change."),
 ("<code>def f(a, b):</code>", "Defines a function: a named, reusable piece of code. <code>return</code> hands its answer back."),
 ("<code>for x in items:</code>", "Repeats the indented lines once for each item."),
 ("<code>if ... / elif ... / else:</code>", "Runs code only when a condition is true."),
 ("<code>[a, b]</code>, <code>{key: value}</code>, <code>(a, b)</code>", "A list, a dictionary (look-up table) and a tuple (a fixed group of values)."),
 ("<code>f\"{x} dogs\"</code>", "Text with values filled in: if x is 3, this is \"3 dogs\"."),
 ("<code>np.array</code> / numpy", "Fast arrays of numbers. An image is a 3-D array: height x width x 3 colours."),
 ("<code>pd.DataFrame</code> / pandas", "A table in memory, like a spreadsheet. <code>df.groupby(\"camera\")</code> splits it into one group per camera; <code>df[df.x &gt; 0]</code> keeps rows where column x is above 0."),
 ("<code>class Name:</code>", "A blueprint bundling data with the functions that use it (e.g. <code>ClockReader</code>)."),
 ("<code>with ...:</code>", "Opens something (a file, a worker pool) and closes it automatically at the end."),
 ("<code>if __name__ == \"__main__\":</code>", "'Run this part only when the file is run directly', not when another file imports it."),
]


def code_block(lines, a, b):
    rows = []
    for n in range(a, b + 1):
        rows.append(f'<span class="ln">{n}</span>{html.escape(lines[n - 1]) or " "}')
    return "\n".join(rows)


def build():
    parts = []
    toc = []
    for i, (path, purpose, n_expected, notes) in enumerate(FILES, 1):
        lines = (ROOT / path).read_text().splitlines()
        if len(lines) != n_expected:
            sys.exit(f"{path} has {len(lines)} lines, notes expect {n_expected}: update the notes")
        covered = set()
        for a, b, _ in notes:
            covered.update(range(a, b + 1))
        missing = [k for k in range(1, len(lines) + 1) if k not in covered and lines[k - 1].strip()]
        if missing:
            sys.exit(f"{path}: non-blank lines without a note: {missing}")
        toc.append(f'<li><span class="mono">{path}</span> <span class="muted">({len(lines)} lines)</span></li>')
        chunks = []
        for a, b, text in notes:
            chunks.append(f'<div class="chunk"><pre>{code_block(lines, a, b)}</pre><p>{text}</p></div>')
        parts.append(f'<section class="file"><h2><span class="n">{i}</span><span class="mono">{path}</span></h2>'
                     f'<p class="purpose">{purpose}</p>{"".join(chunks)}</section>')
    lt = "".join(f'<div class="chunk"><p><b>{h}</b>. {t}</p></div>' for h, t in LABEL_TOOL)
    parts.append(f'<section class="file"><h2><span class="n">{len(FILES) + 1}</span><span class="mono">tools/label_tool.html</span></h2>'
                 f'<p class="purpose">The labelling page you used. It is a web page (HTML, CSS and JavaScript), not part of the Python pipeline, so it is explained block by block rather than line by line.</p>{lt}</section>')
    basics = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in INTRO_BASICS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Code Walkthrough</title>
<style>
:root{{--ink:#1c2420;--muted:#5b675f;--line:#d6ddd8;--soft:#f1f5f2;--code:#f6f7f5;--accent:#1f7a5a;--sans:"Inter","DejaVu Sans",sans-serif;--mono:"DejaVu Sans Mono",monospace}}
@page{{size:A4;margin:14mm 13mm 16mm}}
body{{margin:0;font-family:var(--sans);color:var(--ink);font-size:9.8pt;line-height:1.45;background:#fff}}
h1{{font-size:22pt;margin:0 0 4pt;line-height:1.15}}
h2{{font-size:13pt;margin:0 0 4pt;padding-top:6pt;border-top:1.2pt solid var(--ink);break-after:avoid}}
h2 .n{{color:var(--accent);margin-right:7pt}}
h3{{font-size:11pt;margin:12pt 0 4pt}}
.mono{{font-family:var(--mono);font-size:.92em}} .muted{{color:var(--muted)}}
.lead{{color:var(--muted);font-size:11pt;margin:0 0 10pt}}
.file{{margin-top:16pt}}
.purpose{{background:var(--soft);border-left:3pt solid var(--accent);padding:6pt 9pt;margin:4pt 0 9pt}}
.chunk{{break-inside:avoid;margin:0 0 7pt;padding-bottom:6pt;border-bottom:.5pt solid var(--line)}}
pre{{margin:0 0 3pt;background:var(--code);border-radius:2pt;padding:4pt 6pt;font-family:var(--mono);font-size:7.3pt;line-height:1.4;white-space:pre-wrap;word-break:break-all}}
.ln{{display:inline-block;width:20pt;color:#9aa59f;user-select:none}}
.chunk p{{margin:0}}
code{{font-family:var(--mono);font-size:.88em;background:var(--code);padding:0 2pt;border-radius:2pt}}
table{{border-collapse:collapse;width:100%;font-size:9pt;margin:4pt 0 10pt}}
td,th{{text-align:left;vertical-align:top;padding:3pt 6pt;border-bottom:.5pt solid var(--line)}}
td:first-child{{width:38%}}
ol li,ul li{{margin-bottom:2pt}}
</style></head><body>
<div class="kicker" style="font-size:8pt;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:700">Stray dog detection · code walkthrough</div>
<h1>What every part of the code does</h1>
<p class="lead">Every pipeline file is shown in small groups of lines, each followed by a plain-English explanation. Grey numbers are line numbers, so you can find the same spot in the repository.</p>
<h3>Order the files run in</h3>
<ol>
<li><span class="mono">scripts/download_data.sh</span>: download and unzip the footage</li>
<li><span class="mono">train_clock_reader.py</span>, then <span class="mono">prepare.py</span>: real recording times, duplicates, the manifest</li>
<li><span class="mono">detect.py</span>: YOLO on 3 frames per second (uses <span class="mono">video.py</span>)</li>
<li><span class="mono">track.py</span>, then <span class="mono">stats.py</span>: tracks, visits, activity, charts</li>
<li><span class="mono">select_eval_frames.py</span> → you label with <span class="mono">label_tool.html</span> → <span class="mono">import_labels.py</span> → <span class="mono">evaluate.py</span>: accuracy</li>
<li><span class="mono">tune_tracker.py</span>: tracker settings for the final detector, on dev clips</li>
<li><span class="mono">examples.py</span> and <span class="mono">reid_demo.py</span>: gallery and recognition demo</li>
</ol>
<p><span class="mono">scripts/run_all.sh</span> runs all of this in order with one command.</p>
<p><span class="mono">config.py</span> (settings), <span class="mono">clock_ocr.py</span> (the clock reader) and <span class="mono">boxes.py</span> (box tools) are used by the others.</p>
<h3>Python in 2 minutes</h3>
<table>{basics}</table>
<h3>Files</h3><ul>{"".join(toc)}<li><span class="mono">tools/label_tool.html</span> <span class="muted">(explained by block)</span></li></ul>
{"".join(parts)}
</body></html>"""


if __name__ == "__main__":
    out_html = ROOT / "docs" / "code_walkthrough.html"
    out_html.write_text(build())
    js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://{out_html}',{{waitUntil:'networkidle'}});
await p.pdf({{path:'{ROOT / "docs" / "code_walkthrough.pdf"}',format:'A4',printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',
footerTemplate:'<div style="font-size:7pt;color:#777;width:100%;text-align:center;font-family:Inter">Code walkthrough · <span class=pageNumber></span>/<span class=totalPages></span></div>',
margin:{{top:'14mm',bottom:'16mm',left:'13mm',right:'13mm'}}}});await b.close();}})()"""
    subprocess.run(["node", "-e", js], check=True)
    print("wrote docs/code_walkthrough.pdf")
