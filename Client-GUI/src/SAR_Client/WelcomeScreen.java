package SAR_Client;

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

public class WelcomeScreen {

    static Stage window;

    public static void display(Stage primaryStage) throws Exception {
        window = primaryStage;
        Parent root = FXMLLoader.load(WelcomeScreen.class.getResource("WelcomeScreen.fxml"));
        primaryStage.getIcons().add(new Image("icon.png"));
        primaryStage.setTitle("Welcome to SAR Mission Control");
        Scene scene = new Scene(root, 600, 500);
        scene.getStylesheets().add("WelcomeStyle.css");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public void OnRemoteBtnPressed() throws Exception {
        RemoteClient.Display();
        window.close();
    }

    public void OnDroneBtnPressed() throws Exception {
        DroneProcessingClient.Display();
        window.close();
    }

    public void OnTestBtnPressed() throws Exception {
        TestInterface.Display();
        window.close();
    }
}
