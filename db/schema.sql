-- DermAI Diagnostic — PostgreSQL Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE model_version (
    model_version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    architecture     VARCHAR(100) NOT NULL,
    version_number   VARCHAR(20)  NOT NULL,
    accuracy         DECIMAL(5,4),
    summary          TEXT,
    deployed_at      TIMESTAMP DEFAULT NOW(),
    is_active        BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE doctor (
    doctor_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       VARCHAR(200) NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20) DEFAULT 'doctor' CHECK (role IN ('admin', 'doctor', 'viewer')),
    is_active       BOOLEAN DEFAULT TRUE,
    last_login      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE patient (
    patient_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name         VARCHAR(200) NOT NULL,
    date_of_birth     DATE,
    gender            VARCHAR(10) CHECK (gender IN ('male', 'female', 'other')),
    medical_record_id VARCHAR(50) UNIQUE,
    doctor_id         UUID REFERENCES doctor(doctor_id) ON DELETE SET NULL,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE skin_lesion_image (
    image_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id            UUID REFERENCES patient(patient_id) ON DELETE CASCADE,
    doctor_id             UUID REFERENCES doctor(doctor_id) ON DELETE SET NULL,
    file_path             VARCHAR(500) NOT NULL,
    file_format           VARCHAR(10) DEFAULT 'jpg',
    file_size_kb          INTEGER,
    original_filename     VARCHAR(255),
    uploaded_at           TIMESTAMP DEFAULT NOW(),
    preprocessing_status  VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE diagnosis_report (
    report_id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id                 UUID REFERENCES skin_lesion_image(image_id) ON DELETE CASCADE,
    model_version_id         UUID REFERENCES model_version(model_version_id),
    prediction_class         VARCHAR(10) NOT NULL,
    confidence_score         DECIMAL(5,4) NOT NULL,
    probability_distribution JSONB NOT NULL,
    gradcam_map_path         VARCHAR(500),
    processing_time_ms       INTEGER,
    created_at               TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_session (
    session_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doctor_id     UUID REFERENCES doctor(doctor_id) ON DELETE CASCADE,
    refresh_token VARCHAR(500) UNIQUE NOT NULL,
    expires_at    TIMESTAMP NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_skin_lesion_image_doctor  ON skin_lesion_image(doctor_id);
CREATE INDEX idx_skin_lesion_image_patient ON skin_lesion_image(patient_id);
CREATE INDEX idx_diagnosis_report_image    ON diagnosis_report(image_id);
CREATE INDEX idx_patient_doctor            ON patient(doctor_id);
CREATE INDEX idx_doctor_email             ON doctor(email);

INSERT INTO model_version (architecture, version_number, accuracy, summary, is_active) VALUES
('MobileNetV2', 'v4.0', 0.8082,
 'Transfer Learning на HAM10000. Oversampling + обратные веса классов. Accuracy 80.8%.',
 TRUE);

-- Пароль: admin123
INSERT INTO doctor (full_name, email, password_hash, role) VALUES
('Администратор', 'admin@dermai.kz', '$2b$12$h7jJjOWxDy2.2CZW/Un4lusAj1qes0L0E9UtBJvNjQ9HAbfmXKMz2', 'admin');
