package SAR_Client;

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

import java.io.IOException;

public class RemoteClient extends Pipeline {

    public static void Display() throws Exception {
        Stage window;
        window = new Stage();
        Parent root = FXMLLoader.load(RemoteClient.class.getResource("RemoteClient.fxml"));
        window.getIcons().add(new Image("icon.png"));
        window.setTitle("Remote Processing Client");
        Scene scene = new Scene(root, 600, 500);
        window.setScene(scene);
        window.show();
    }

    public void OnPreviewBtnPressed()
    {
        
    }

    public void OnPipelineBtnPressed()
    {

    }

    void RunDetection() throws IOException
    {

    }

    void RunHAR() throws IOException
    {

    }

    void RunDetermination() throws IOException
    {

    }
}
