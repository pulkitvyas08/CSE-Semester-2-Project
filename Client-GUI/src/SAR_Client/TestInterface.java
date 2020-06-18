package SAR_Client;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Group;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.scene.media.Media;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Scanner;

public class TestInterface extends Pipeline {

    static Stage window;
    File selectedVideo;
    boolean videoLoaded = false, pipelineExecuted = false;
    int fileCounter = 1;
    int frameCounter = 0;
    boolean helpNeeded;

    public static void Display() throws Exception {
        window = new Stage();
        Parent root = FXMLLoader.load(TestInterface.class.getResource("TestScreen.fxml"));
        window.getIcons().add(new Image("icon.png"));
        window.setTitle("Test Interface");
        Scene scene = new Scene(root, 600, 500);
        window.setScene(scene);
        window.show();
    }

    public void ResultButtonClicked() {
        if(!pipelineExecuted || !videoLoaded)
            System.exit(0);
        DetectionResults.display(frameCounter, helpNeeded);
    }

    public void OnVideoBtnPressed() {
        Stage newWindow = new Stage();

        FileChooser fileChooser = new FileChooser();
        selectedVideo = fileChooser.showOpenDialog(window);

        Media media = new Media(new File(selectedVideo.getAbsolutePath()).toURI().toString());
        MediaPlayer mediaPlayer = new MediaPlayer(media);
        MediaView mediaView = new MediaView(mediaPlayer);
        mediaPlayer.setAutoPlay(true);

        Group root = new Group();
        root.getChildren().add(mediaView);
        Scene scene = new Scene(root, 500, 400);
        newWindow.setScene(scene);

        mediaView.setFitWidth(newWindow.getWidth());
        mediaView.setFitHeight(newWindow.getHeight());

        newWindow.getIcons().add(new Image("icon.png"));
        newWindow.setTitle("Playing video");
        newWindow.show();
        videoLoaded = true;
    }

    public void OnPipelineBtnPressed() {
        if(selectedVideo == null)
            System.exit(0);

        /* Pipeline: button pressed -> convert video to images -> loop over images and save the prediction bounding
                   boxes -> loop over the bounding boxes and apply HAR -> run determination and output final result */

        // convert video to images
        try {
            splitInFrames();
        }catch(IOException exc)
        {
            System.out.println(exc.getMessage());
        }

        // loop over images and save the prediction bounding boxes
        try {
            RunDetection();
        }catch(IOException exc)
        {
            System.out.println(exc.getMessage());
        }

        // looping over bounding boxes and applying HAR
        try {
            RunHAR();
        }catch (IOException exc)
        {
            System.out.println(exc.getMessage());
        }

        // run determination
        try {
            RunDetermination();
        }catch(IOException exc)
        {
            System.out.println(exc.getMessage());
        }

        // Alert for pipeline execution completion
        GeneralAlert.display("Pipeline Executed", "Pipeline execution completed successfully, click on" +
                " results to load results for this iteration");

        pipelineExecuted = true;
    }

    void RunDetection() throws IOException {

        File[] files = new File("video_frames/").listFiles();
        for(File file : files)
        {
            Runtime.getRuntime().exec(
                    "Darknet-Module/./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights " +
                    file.getAbsolutePath());
            move_roi();
        }
    }

    void RunHAR() throws IOException {
        File[] files = new File("Darknet-Module/data/all_cropped_roi/").listFiles();
        for(File file: files)
        {
            Runtime.getRuntime().exec("python Human-AC-Module/run_image.py --image=" + file.getAbsolutePath() +
                    ".jpg");
            frameCounter++;
        }
    }

    void RunDetermination() throws IOException {
        final int confidencePercentage = 35;

        int numberOfPoses = 0;
        int numberOfWaves = 0;

        try {
            Scanner scanner = new Scanner(new File("/Users/pankaj/Downloads/myfile.txt"));
            while (scanner.hasNextLine()) {
                if(scanner.nextLine() == "waving")
                    numberOfWaves++;
                numberOfPoses++;
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        double predictionPercent = ((double)numberOfWaves/(double)numberOfPoses) * 100;

        if(confidencePercentage >= predictionPercent)
            helpNeeded = true;
        else
            helpNeeded = false;

        // reset counters and flags
        fileCounter = 1;
        frameCounter = 0;
        pipelineExecuted = false;
        videoLoaded = false;

        // delete files
        Arrays.stream(new File("video_frames/").listFiles()).forEach(File::delete);
        Arrays.stream(new File("Darknet-Module/all_cropped_roi/").listFiles()).forEach(File::delete);
        Files.deleteIfExists(Paths.get("results.txt"));
        File newResults = new File("results.txt");
    }

    private void splitInFrames() throws IOException {
        Runtime.getRuntime().exec("python video_frames.py " + selectedVideo.getAbsolutePath());
    }

    private void move_roi() throws IOException{
        File[] files = new File("Darknet-Module/data/cropped_roi/").listFiles();
        for(File file : files)
        {
            Files.move(Paths.get(file.getAbsolutePath()), Paths.get("Darknet-Module/data/all_cropped_roi/" +
                    fileCounter++ + ".jpg"));
        }

        // delete all files from this iteration
        Arrays.stream(new File("Darknet-Module/cropped_roi/").listFiles()).forEach(File::delete);
    }
}
