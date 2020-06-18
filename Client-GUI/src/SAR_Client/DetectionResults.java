package SAR_Client;

import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

public class DetectionResults {

    public static void display(int frameCounter, boolean helpNeeded) {

        Stage window = new Stage();

        window.initModality(Modality.APPLICATION_MODAL);
        window.setTitle("Detection Results");
        window.setMinWidth(250);
        Label label = new Label();

        String decision;
        if(helpNeeded)
            decision = "Help";
        else
            decision = "No Help";

        label.setText("Taking average of " + frameCounter + " frames system has determined: " + decision);

        Button btn = new Button("close");

        VBox layout = new VBox();
        layout.getChildren().addAll(label, btn);
        layout.setAlignment(Pos.CENTER);
        Scene scene = new Scene(layout);
        window.getIcons().add(new Image("icon.png"));
        window.setScene(scene);
        window.showAndWait();
    }
}
