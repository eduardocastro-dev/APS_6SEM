module br.aps.aps_6sem {
    requires javafx.controls;
    requires javafx.fxml;
    requires opencv;
    requires java.logging;


    opens br.aps.aps_6sem to javafx.fxml;
    exports br.aps.aps_6sem;
}