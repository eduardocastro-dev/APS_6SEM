
package br.aps.aps_6sem;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class WebcamFaceDetectionFX extends Application {

    private static final Logger LOGGER = Logger.getLogger(WebcamFaceDetectionFX.class.getName());
    private VideoCapture capture;
    private ImageView imageView;
    private CascadeClassifier faceDetector;
    private boolean isRunning = false;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        setupLogger();
    }

    private static void setupLogger() {
        try {
            FileHandler fileHandler = new FileHandler("WebcamFaceDetection.log", true);
            SimpleFormatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            LOGGER.addHandler(fileHandler);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to set up logger", e);
        }
    }

    @Override
    public void start(Stage primaryStage) {
        imageView = new ImageView();
        imageView.setFitWidth(640);
        imageView.setFitHeight(480);

        Button startButton = new Button("Iniciar/Parar Webcam");
        startButton.setOnAction(e -> toggleWebcam());

        VBox root = new VBox(10, imageView, startButton);
        Scene scene = new Scene(root);

        primaryStage.setTitle("Detecção de Rosto com Webcam");
        primaryStage.setScene(scene);
        primaryStage.show();

        capture = new VideoCapture(0);
        faceDetector = new CascadeClassifier();
        String cascadePath = "C:\\WS-JetBrains\\aps_6sem\\Cascade\\haarcascade_frontalcatface.xml";

        File cascadeFile = new File(cascadePath);
        if (!cascadeFile.exists()) {
            LOGGER.severe("Erro: O arquivo do classificador não existe no caminho especificado.");
            LOGGER.severe("Caminho procurado: " + cascadeFile.getAbsolutePath());
            Platform.exit();
            return;
        }

        if (!faceDetector.load(cascadePath)) {
            LOGGER.severe("Erro ao carregar o classificador Haar Cascade.");
            LOGGER.severe("Verifique se o caminho está correto e se o arquivo não está corrompido.");
            Platform.exit();
            return;
        }

        LOGGER.info("Classificador Haar Cascade carregado com sucesso.");
    }

    private void toggleWebcam() {
        if (isRunning) {
            isRunning = false;
        } else {
            isRunning = true;
            new Thread(this::processWebcam).start();
        }
    }

    private void processWebcam() {
        Mat frame = new Mat();
        while (isRunning) {
            capture.read(frame);
            if (!frame.empty()) {
                MatOfRect faceDetections = new MatOfRect();
                try {
                    faceDetector.detectMultiScale(frame, faceDetections);

                    for (Rect rect : faceDetections.toArray()) {
                        Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                                new Point(rect.x + rect.width, rect.y + rect.height),
                                new Scalar(0, 255, 0), 2);
                    }

                    Image image = mat2Image(frame);
                    Platform.runLater(() -> imageView.setImage(image));
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Erro durante a detecção de rosto", e);
                }
            }
        }
    }

    private Image mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

    @Override
    public void stop() {
        isRunning = false;
        if (capture != null) {
            capture.release();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}