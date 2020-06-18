package SAR_Client;

import java.io.IOException;

public abstract class Pipeline {
    abstract void RunDetection() throws IOException;
    abstract void RunHAR() throws IOException;
    abstract void RunDetermination() throws IOException;
}
