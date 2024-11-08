package br.aps.aps_6sem;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class WebcamFaceDetectionFX extends Application {

    private static final Logger LOGGER = Logger.getLogger(WebcamFaceDetectionFX.class.getName());
    private static final String IMAGES_FOLDER = "imagens_rosto"; // Base folder for user images
    private static final String LIVE_FOLDER = "img_temp"; // Folder for live face capture
    private static final int CAPTURE_RATE = 1; // 1 second per frame (just for demonstration)
    private static final int CAPTURE_TIME = 10; // 30 seconds for face capture
    private static final int MAX_LOGIN_CAPTURES = 5; // Number of images to capture for login
    private static final int TOTAL_REGISTRATION_CAPTURES = 60; // Total captures for registration
    private static final double SIMILARITY_THRESHOLD = 0.3; // 20% similarity threshold
    private static final int CLEANUP_INTERVAL = 30; // Clean up 'img_temp' folder every 30 seconds

    private VideoCapture capture;
    private ImageView imageView;
    private CascadeClassifier faceDetector;
    private boolean isRunning = false;
    private Map<String, String> userMap = new HashMap<>(); // Stores user info (name_permission_hash)
    private String currentUserName; // Stores the current user's name
    private int captureCount = 0;
    private long startTime;
    private long lastCleanupTime = 0; // Track the last time the folder was cleaned

    // Variável global para armazenar as informações do usuário
    private String loggedUserName; // Nome do usuário logado
    private String loggedUserPermission; // Permissão do usuário logado

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        setupLogger();
    }

    private static void setupLogger() {
        try {
            FileHandler fileHandler = new FileHandler("FaceDetectionApp.log", true);
            SimpleFormatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            LOGGER.addHandler(fileHandler);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to set up logger", e);
        }
    }

    @Override
    public void start(Stage primaryStage) {
        // Main screen with "Register" and "Login" buttons
        Button registerButton = new Button("Cadastro");
        Button loginButton = new Button("Login");

        registerButton.setOnAction(e -> showRegistrationScreen(primaryStage));
        loginButton.setOnAction(e -> showLoginScreen(primaryStage));

        VBox root = new VBox(10, registerButton, loginButton);
        Scene scene = new Scene(root);

        primaryStage.setTitle("Face Detection App");
        primaryStage.setScene(scene);
        primaryStage.show();

        // Initialize OpenCV
        capture = new VideoCapture(0);
        faceDetector = new CascadeClassifier();
        String cascadePath = "C:\\WS-JetBrains\\aps_6sem\\Cascade\\haarcascade_frontalface_default.xml";

        File cascadeFile = new File(cascadePath);
        if (!cascadeFile.exists()) {
            LOGGER.severe("Error: Classifier file does not exist at the specified path.");
            LOGGER.severe("Path searched: " + cascadeFile.getAbsolutePath());
            Platform.exit();
            return;
        }

        if (!faceDetector.load(cascadePath)) {
            LOGGER.severe("Error loading Haar Cascade classifier.");
            LOGGER.severe("Check if the path is correct and the file is not corrupted.");
            Platform.exit();
            return;
        }

        LOGGER.info("Haar Cascade classifier loaded successfully.");

        // Start the cleanup thread (runs in the background)
        new Thread(this::cleanupLiveFolder).start();
    }

    private void showRegistrationScreen(Stage primaryStage) {
        // Registration screen
        Label nameLabel = new Label("Nome completo:");
        TextField nameField = new TextField();

        Label permissionLabel = new Label("Nível de permissão:");
        TextField permissionField = new TextField();

        Button registerButton = new Button("Concluir");
        registerButton.setOnAction(e -> {
            String name = nameField.getText();
            String permission = permissionField.getText();
            startFaceCapture(primaryStage, name, permission);
        });

        GridPane registrationPane = new GridPane();
        registrationPane.setVgap(10);
        registrationPane.setHgap(10);
        registrationPane.addRow(0, nameLabel, nameField);
        registrationPane.addRow(1, permissionLabel, permissionField);
        registrationPane.addRow(2, registerButton);

        Scene registrationScene = new Scene(registrationPane);
        primaryStage.setScene(registrationScene);
        primaryStage.show();
    }

    private void startFaceCapture(Stage primaryStage, String name, String permission) {
        // Face capture screen
        imageView = new ImageView();
        imageView.setFitWidth(640);
        imageView.setFitHeight(480);

        Label timerLabel = new Label("Tempo restante: " + CAPTURE_TIME);
        HBox timerBox = new HBox(10, timerLabel);
        timerBox.setAlignment(Pos.CENTER);

        VBox root = new VBox(10, imageView, timerBox);
        root.setPadding(new Insets(20));
        Scene scene = new Scene(root);

        primaryStage.setTitle("Captura de Rosto");
        primaryStage.setScene(scene);
        primaryStage.show();

        startTime = System.currentTimeMillis();
        new Thread(() -> {
            capture = new VideoCapture(0);
            captureCount = 0;
            isRunning = true;
            processWebcam(name, permission, timerLabel, primaryStage); // Pass primaryStage
        }).start();
    }

    private void showLoginScreen(Stage primaryStage) {
        // Login screen
        Label nameLabel = new Label("Nome completo:");
        TextField nameField = new TextField();


        Button loginButton = new Button("Login");
        loginButton.setOnAction(e -> {
            String name = nameField.getText();
            loggedUserName = name;
            // Search for user folder
            File userFolder = new File(IMAGES_FOLDER + File.separator + name);
            if (userFolder.exists() && userFolder.isDirectory()) {
                // User folder found - Start live face capture
                startLiveFaceCapture(primaryStage, name);
            } else {
                // User not found
                LOGGER.info("User not found: " + name);
                showErrorMessage("Usuário não encontrado.");
            }
        });

        GridPane loginPane = new GridPane();
        loginPane.setVgap(10);
        loginPane.setHgap(10);
        loginPane.addRow(0, nameLabel, nameField);
        loginPane.addRow(1, loginButton);

        Scene loginScene = new Scene(loginPane);
        primaryStage.setScene(loginScene);
        primaryStage.show();
    }

    private void startLiveFaceCapture(Stage primaryStage, String name) {
        // Live face capture for login
        imageView = new ImageView();
        imageView.setFitWidth(640);
        imageView.setFitHeight(480);

        // Remove timer label for login
        // Label timerLabel = new Label("Tempo restante: " + CAPTURE_TIME);
        // HBox timerBox = new HBox(10, timerLabel);
        // timerBox.setAlignment(Pos.CENTER);

        VBox root = new VBox(10, imageView); // Remove timerBox
        root.setPadding(new Insets(20));
        Scene scene = new Scene(root);

        primaryStage.setTitle("Login por Reconhecimento Facial");
        primaryStage.setScene(scene);
        primaryStage.show();

        // Create the 'img_temp' folder (if it doesn't exist)
        File imgTempFolder = new File(LIVE_FOLDER);
        if (!imgTempFolder.exists()) {
            imgTempFolder.mkdir();
        }

        startTime = System.currentTimeMillis();
        new Thread(() -> {
            capture = new VideoCapture(0);
            captureCount = 0;
            isRunning = true;
            processLiveFaceCapture(name, primaryStage); // Pass primaryStage
        }).start();
    }

    private void processLiveFaceCapture(String name, Stage primaryStage) {
        Mat frame = new Mat();
        long endTime;

        // Define currentUserName
        currentUserName = name;

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

                        if (captureCount % CAPTURE_RATE == 0 && captureCount < MAX_LOGIN_CAPTURES) {
                            // Capture image every CAPTURE_RATE seconds and limit to MAX_LOGIN_CAPTURES
                            String imagePath = "live_" + captureCount / CAPTURE_RATE + ".jpg";
                            saveLiveImage(frame, imagePath);

                            // Generate and save hash to a .txt file
                            // String hash = getHash(frame);
                            // saveHashToFile("live_" + String.valueOf(captureCount / CAPTURE_RATE), hash); // Correct calling
                        }
                    }

                    Image image = mat2Image(frame);
                    Platform.runLater(() -> imageView.setImage(image));
                    captureCount++;

                    // Update timer
                    endTime = System.currentTimeMillis();
                    long remainingTime = (startTime + (CAPTURE_TIME * 1000)) - endTime;
                    if (remainingTime > 0) {
                        // Remove timer update for login
                        // Platform.runLater(() -> timerLabel.setText("Tempo restante: " + remainingTime / 1000));
                    } else {
                        // Capture ended
                        isRunning = false;
                        Platform.runLater(() -> {
                            capture.release();
                            primaryStage.close();
                            compareLiveImagesWithRegistration(name); // Call the comparison function
                        });
                    }
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Error during face detection", e);
                }
            }
        }
    }

    private void compareLiveImagesWithRegistration(String name) {
        // Get the registered images for the user
        List<Mat> registeredImages = getRegisteredImages(name);

        // Get the live images captured during login
        List<Mat> liveImages = getLiveImages();

        // Calculate the similarity between the live images and the registered images
        double totalSimilarity = 0.0;
        for (Mat liveImage : liveImages) {
            for (Mat registeredImage : registeredImages) {
                totalSimilarity += calculateSimilarity(liveImage, registeredImage);
            }
        }

        // Calculate the average similarity
        double averageSimilarity = totalSimilarity / (liveImages.size() * registeredImages.size());

        // Check if the average similarity is above the threshold
        if (averageSimilarity >= SIMILARITY_THRESHOLD) {
            // Login successful
            showLoginSuccessPopup(name);
        } else {
            // Login failed
            showLoginFailed();
        }
    }

    private List<Mat> getRegisteredImages(String userName) {
        List<Mat> images = new ArrayList<>();
        File userFolder = new File(IMAGES_FOLDER + File.separator + userName);
        if (userFolder.exists() && userFolder.isDirectory()) {
            File[] files = userFolder.listFiles((dir, name) -> name.endsWith(".jpg"));
            if (files != null) {
                for (File file : files) {
                    Mat image = Imgcodecs.imread(file.getAbsolutePath());
                    images.add(image);
                }
            }
        }
        return images;
    }

    private List<Mat> getLiveImages() {
        List<Mat> images = new ArrayList<>();
        File liveFolder = new File(LIVE_FOLDER);
        if (liveFolder.exists() && liveFolder.isDirectory()) {
            File[] files = liveFolder.listFiles((dir, name) -> name.endsWith(".jpg"));
            if (files != null) {
                for (File file : files) {
                    Mat image = Imgcodecs.imread(file.getAbsolutePath());
                    images.add(image);
                }
            }
        }
        return images;
    }

    private double calculateSimilarity(Mat image1, Mat image2) {
        // Convert images to grayscale
        Mat grayImage1 = new Mat();
        Imgproc.cvtColor(image1, grayImage1, Imgproc.COLOR_BGR2GRAY);

        Mat grayImage2 = new Mat();
        Imgproc.cvtColor(image2, grayImage2, Imgproc.COLOR_BGR2GRAY);

        // Calculate the normalized cross-correlation coefficient
        Mat result = new Mat();
        Imgproc.matchTemplate(grayImage1, grayImage2, result, Imgproc.TM_CCOEFF_NORMED);

        // Get the maximum similarity value
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        double similarity = mmr.maxVal;

        return similarity;
    }

    private void showLoginSuccessPopup(String name) {
        // Display a message indicating successful login
        // Get permission from registered user data
        // String userInfo = userMap.get(name); // Não use essa variável aqui
        // String[] parts = userInfo.split("_"); // Não use essa variável aqui
        // String permission = parts[1]; // Não use essa variável aqui

        // Create a new popup stage
        Stage successStage = new Stage();
        successStage.initModality(Modality.APPLICATION_MODAL);
        successStage.setTitle("Login Concluído");

        // Create a label with the success message
        Label successMessage = new Label("Autenticação realizada com sucesso!");
        Label nameLabel = new Label("Nome: " + loggedUserName); // Use a variável global
        Label permissionLabel = new Label("Nível de permissão: " + loggedUserPermission); // Use a variável global

        // Create a button to close the popup
        Button okButton = new Button("OK");
        okButton.setOnAction(e -> successStage.close());

        // Create a VBox to hold the message and button
        VBox successPane = new VBox(10, successMessage, nameLabel, permissionLabel, okButton);
        successPane.setAlignment(Pos.CENTER);
        successPane.setPadding(new Insets(20));

        // Create a scene for the popup
        Scene successScene = new Scene(successPane);

        // Set the scene for the popup stage
        successStage.setScene(successScene);

        // Show the popup
        successStage.showAndWait(); // Use showAndWait to block until the popup is closed
    }

    private void showLoginFailed() {
        // Display a message indicating login failed
        Label errorMessage = new Label("Login falhou. Verifique o nome e tente novamente.");
        Button okButton = new Button("OK");
        okButton.setOnAction(e -> {
            Stage errorStage = (Stage) okButton.getScene().getWindow();
            errorStage.close();
        });

        VBox errorPane = new VBox(10, errorMessage, okButton);
        Scene errorScene = new Scene(errorPane);

        Stage errorStage = new Stage();
        errorStage.setTitle("Erro de Login");
        errorStage.setScene(errorScene);
        errorStage.show();
    }

    private void registerUser(String name, String permission) { // Remove o parâmetro 'hash'
        // String userInfo = name + "_" + permission + "_" + hash; // Não é necessário definir userInfo aqui
        userMap.put(name, name + "_" + permission); // Use o nome e permissão para criar a chave do mapa
        LOGGER.info("Registered user: " + name + ", Permission: " + permission);
    }

    private void showUserInfo(String name, String permission) {
        // Display user information on a new screen
        Label nameLabel = new Label("Nome: " + name);
        Label permissionLabel = new Label("Nível de permissão: " + permission);

        VBox userInfoPane = new VBox(10, nameLabel, permissionLabel);
        Scene userInfoScene = new Scene(userInfoPane);

        Stage userInfoStage = new Stage();
        userInfoStage.setTitle("Informações do Usuário");
        userInfoStage.setScene(userInfoScene);
        userInfoStage.show();
    }

    private void processWebcam(String name, String permission, Label timerLabel, Stage primaryStage) {
        Mat frame = new Mat();
        long endTime;

        // Create user folder INSIDE imagens_rosto
        File userFolder = new File(IMAGES_FOLDER + File.separator + name);
        if (!userFolder.exists()) {
            userFolder.mkdir();
        }

        // Define currentUserName
        currentUserName = name;

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

                        // Gere o hash para cada imagem capturada, independentemente do CAPTURE_RATE
                        // String hash = getHash(frame);

                        if (captureCount % CAPTURE_RATE == 0 && captureCount < TOTAL_REGISTRATION_CAPTURES) {
                            // Capture image every CAPTURE_RATE seconds
                            String imagePath = name + File.separator + name + "_" + captureCount / CAPTURE_RATE + ".jpg";
                            saveImage(frame, imagePath);

                            // Salve o hash no arquivo na pasta do usuário
                            // saveHashToFile(name + "_" + captureCount / CAPTURE_RATE, hash); // Nomeie o arquivo com o contador
                        }
                    }

                    Image image = mat2Image(frame);
                    Platform.runLater(() -> imageView.setImage(image));
                    captureCount++;

                    // Update timer
                    endTime = System.currentTimeMillis();
                    long remainingTime = (startTime + (CAPTURE_TIME * 1000)) - endTime;
                    if (remainingTime > 0) {
                        Platform.runLater(() -> timerLabel.setText("Tempo restante: " + remainingTime / 1000));
                    } else {
                        // Capture ended
                        isRunning = false;
                        Platform.runLater(() -> {
                            capture.release();
                            primaryStage.close();
                            //showRegistrationSuccessMessage(name, permission);
                        });
                    }
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Error during face detection", e);
                }
            }
        }
    }

    private void showRegistrationSuccessMessage(String name, String permission) {
        // Display a message indicating successful registration
        Label successMessage = new Label("Cadastro concluído com sucesso!");
        Button okButton = new Button("OK");
        okButton.setOnAction(e -> {
            Stage successStage = (Stage) okButton.getScene().getWindow();
            successStage.close();

        });

        VBox successPane = new VBox(10, successMessage, okButton);
        Scene successScene = new Scene(successPane);

        Stage successStage = new Stage();
        successStage.setTitle("Cadastro Concluído");
        successStage.setScene(successScene);
        successStage.show();
    }

    private void showErrorMessage(String message) {
        // Display an error message
        Label errorMessage = new Label(message);
        Button okButton = new Button("OK");
        okButton.setOnAction(e -> {
            Stage errorStage = (Stage) okButton.getScene().getWindow();
            errorStage.close();
        });

        VBox errorPane = new VBox(10, errorMessage, okButton);
        Scene errorScene = new Scene(errorPane);

        Stage errorStage = new Stage();
        errorStage.setTitle("Erro");
        errorStage.setScene(errorScene);
        errorStage.show();
    }

    private String getHashFromWebcam() {
        // Capture a frame from the webcam and generate a hash
        Mat frame = new Mat();
        capture.read(frame);
        return getHash(frame);
    }

    private String getHash(Mat frame) {
        try {
            // Generate a hash from the face features
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] messageDigest = md.digest(frame.toString().getBytes());
            BigInteger no = new BigInteger(1, messageDigest);
            String hashtext = no.toString(16);
            return hashtext;
        } catch (NoSuchAlgorithmException e) {
            LOGGER.log(Level.SEVERE, "Error generating hash", e);
            return "";
        }
    }

    private void saveImage(Mat frame, String imagePath) {
        // Save the captured face image to the file system
        try {
            MatOfByte buffer = new MatOfByte();
            Imgcodecs.imencode(".jpg", frame, buffer);
            byte[] bytes = buffer.toArray();
            Files.write(Paths.get(IMAGES_FOLDER + File.separator + imagePath), bytes); // Combine with IMAGES_FOLDER
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error saving image", e);
        }
    }

    private void saveLiveImage(Mat frame, String imagePath) {
        // Save the captured face image to the file system
        try {
            MatOfByte buffer = new MatOfByte();
            Imgcodecs.imencode(".jpg", frame, buffer);
            byte[] bytes = buffer.toArray();
            Files.write(Paths.get(LIVE_FOLDER + File.separator + imagePath), bytes); // Combine with LIVE_FOLDER
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error saving image", e);
        }
    }

    private void saveHashToFile(String fileName, String hash) {
        // Save the hash to a .txt file in the user folder
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(LIVE_FOLDER + File.separator + fileName + ".txt"))) {
            writer.write(hash);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error saving hash to file", e);
        }
    }

    private String getHashFromUserFolder(String userName) {
        // Get the hash from the last .txt file in the user folder
        File userFolder = new File(IMAGES_FOLDER + File.separator + userName);
        if (userFolder.exists() && userFolder.isDirectory()) {
            File[] files = userFolder.listFiles((dir, name) -> name.endsWith(".txt"));
            if (files != null && files.length > 0) {
                File lastFile = files[files.length - 1];
                try {
                    return Files.readString(Paths.get(lastFile.getAbsolutePath()));
                } catch (IOException e) {
                    LOGGER.log(Level.SEVERE, "Error reading hash from file", e);
                }
            }
        }
        return "";
    }

    private boolean isSimilarToRegisteredHashes(String userName, String currentHash) {
        List<String> registeredHashes = getHashesFromUserFolder(userName);
        if (registeredHashes.isEmpty()) {
            return false; // No registered hashes to compare
        }

        // Compare currentHash with all registered hashes
        for (String registeredHash : registeredHashes) {
            if (isSimilarHashes(currentHash, registeredHash)) {
                return true; // Login successful if at least one hash matches
            }
        }

        return false; // No similar hashes found
    }

    private List<String> getHashesFromUserFolder(String userName) {
        List<String> hashes = new ArrayList<>();
        File userFolder = new File(IMAGES_FOLDER + File.separator + userName); // Use user folder
        if (userFolder.exists() && userFolder.isDirectory()) {
            File[] files = userFolder.listFiles((dir, name) -> name.endsWith(".txt"));
            if (files != null) {
                for (File file : files) {
                    try {
                        String hash = Files.readString(Paths.get(file.getAbsolutePath()));
                        hashes.add(hash);
                    } catch (IOException e) {
                        LOGGER.log(Level.SEVERE, "Error reading hash from file", e);
                    }
                }
            }
        }
        return hashes;
    }

    private boolean isSimilarHashes(String hash1, String hash2) {
        // Calculate Hamming distance (number of differing bits)
        int hammingDistance = 0;
        for (int i = 0; i < hash1.length(); i++) {
            if (hash1.charAt(i) != hash2.charAt(i)) {
                hammingDistance++;
            }
        }

        // Calculate similarity based on Hamming distance
        double similarity = 1 - ((double) hammingDistance / hash1.length());

        // Compare similarity against threshold
        return similarity >= SIMILARITY_THRESHOLD;
    }

    private Image mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

    private void cleanupLiveFolder() {
        while (isRunning) {
            try {
                Thread.sleep(CLEANUP_INTERVAL * 1000); // Sleep for CLEANUP_INTERVAL seconds

                // Check if enough time has passed since the last cleanup
                if (System.currentTimeMillis() - lastCleanupTime >= CLEANUP_INTERVAL * 1000) {
                    File liveFolder = new File(LIVE_FOLDER);
                    if (liveFolder.exists() && liveFolder.isDirectory()) {
                        File[] files = liveFolder.listFiles();
                        if (files != null) {
                            for (File file : files) {
                                if (file.isFile()) {
                                    file.delete();
                                }
                            }
                        }
                    }
                    lastCleanupTime = System.currentTimeMillis(); // Update last cleanup time
                }
            } catch (InterruptedException e) {
                LOGGER.log(Level.WARNING, "Cleanup thread interrupted", e);
                Thread.currentThread().interrupt(); // Re-interrupt the thread
            }
        }
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