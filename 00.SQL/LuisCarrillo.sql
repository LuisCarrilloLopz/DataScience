###############################################
###### UNIVERSIDAD COMPLUTENSE DE MADRID ######
######  MASTER EN BIG DATA, DATA SCIENCE ######
######	    INTELIGENCIA ARTIFICIAL      ######
######    LUIS EDUARDO CARRILLO LOPEZ.   ######
###############################################

-- Crear una nueva base de datos llamada ArteVida (si ya existe, la elimina).
DROP DATABASE IF EXISTS ArteVida;
CREATE DATABASE ArteVida;
USE ArteVida;

-- Crear la tabla ACTIVIDAD para almacenar información sobre actividades.
CREATE TABLE ACTIVIDAD (
    ID_ACTIVIDAD VARCHAR(255) PRIMARY KEY, -- Clave primaria para identificar la actividad.
    TP_ACTIVIDAD VARCHAR(255), -- Tipo de actividad.
    DURACION DECIMAL(10, 2), -- Duración de la actividad.
    DESCRIPCION VARCHAR(255) -- Descripción de la actividad.
);

-- Crear la tabla ARTISTA para almacenar información sobre artistas.
CREATE TABLE ARTISTA (
    ID_ARTISTA VARCHAR(255) PRIMARY KEY, -- Clave primaria para identificar al artista.
    NOMBRE VARCHAR(255), -- Nombre del artista.
    COSTO DECIMAL(10, 2), -- Costo del artista por su participación en una actividad.
    BIOGRAFIA TEXT, -- Biografía del artista.
    GENERO VARCHAR(255) -- Género artístico del artista.
);

-- Crear la tabla UBICACION para almacenar información sobre ubicaciones.
CREATE TABLE UBICACION (
    ID_DIRECCION VARCHAR(255) PRIMARY KEY, -- Clave primaria para identificar la ubicación.
    CALLE VARCHAR(255), -- Nombre de la calle.
    CIUDAD VARCHAR(255), -- Nombre de la ciudad.
    PUEBLO VARCHAR(255), -- Nombre del pueblo (si está disponible).
    AFORO INT, -- Capacidad máxima de asistentes en la ubicación.
    ALQUILER DECIMAL(10, 2), -- Costo de alquiler de la ubicación.
    CARACTERISTICAS TEXT, -- Características especiales de la ubicación.
    CONTACTO VARCHAR(255), -- Persona de contacto para la ubicación.
    TELEFONO VARCHAR(20) -- Número de teléfono de contacto.
);

-- Crear la tabla PERSONAS para almacenar información sobre personas.
CREATE TABLE PERSONAS (
    ID_PERSONAS VARCHAR(255) PRIMARY KEY, -- Clave primaria para identificar a la persona.
    NOMBRE VARCHAR(255), -- Nombre de la persona.
    E_MAIL VARCHAR(255) -- Dirección de correo electrónico de la persona.
);

-- Crear la tabla CONTACTO para almacenar información de contacto de personas.
CREATE TABLE CONTACTO (
    TELEFONO VARCHAR(20) PRIMARY KEY, -- Clave primaria para identificar el teléfono.
    NOMBRE VARCHAR(255), -- Nombre de la persona.
    PAIS_LADA VARCHAR(255), -- Prefijo de país y LADA del teléfono.
    ID_PERSONAS VARCHAR(255), -- Clave foránea que se relaciona con la tabla PERSONAS.
    FOREIGN KEY (ID_PERSONAS) REFERENCES PERSONAS(ID_PERSONAS) -- Restricción de clave foránea.
);

-- Crear la tabla EVENTO para almacenar información sobre eventos.
CREATE TABLE EVENTO (
    ID_EVENTO VARCHAR(255) PRIMARY KEY, -- Clave primaria para identificar el evento.
    NOMBRE VARCHAR(255), -- Nombre del evento.
    FECHA DATE, -- Fecha del evento.
    HORA TIME, -- Hora del evento.
    DESCRIPCION TEXT, -- Descripción del evento.
    PRECIO DECIMAL(10, 2), -- Precio de entrada al evento.
    ID_ACTIVIDAD VARCHAR(255), -- Clave foránea que se relaciona con la tabla ACTIVIDAD.
    ID_DIRECCION VARCHAR(255), -- Clave foránea que se relaciona con la tabla UBICACION.
    INGRESO_MINIMO DECIMAL(10, 2), -- Ingreso mínimo requerido para el evento.
    FOREIGN KEY (ID_ACTIVIDAD) REFERENCES Actividad(ID_ACTIVIDAD), -- Restricción de clave foránea.
    FOREIGN KEY (ID_DIRECCION) REFERENCES UBICACION(ID_DIRECCION) -- Restricción de clave foránea.
);

-- Crear la tabla que relaciona PERSONAS con EVENTO.
CREATE TABLE PERSONA_EVENTO (
    ID_PERSONAS VARCHAR(255), -- Clave foránea que se relaciona con la tabla PERSONAS.
    ID_EVENTO VARCHAR(255), -- Clave foránea que se relaciona con la tabla EVENTO.
    PRIMARY KEY (ID_PERSONAS, ID_EVENTO), -- Clave primaria compuesta.
    FOREIGN KEY (ID_PERSONAS) REFERENCES PERSONAS(ID_PERSONAS), -- Restricción de clave foránea.
    FOREIGN KEY (ID_EVENTO) REFERENCES EVENTO(ID_EVENTO) -- Restricción de clave foránea.
);

-- Crear la tabla que relaciona ACTIVIDAD con ARTISTA.
CREATE TABLE ACTIVIDAD_ARTISTA (
    ID_ACTIVIDAD VARCHAR(255), -- Clave foránea que se relaciona con la tabla ACTIVIDAD.
    ID_ARTISTA VARCHAR(255), -- Clave foránea que se relaciona con la tabla ARTISTA.
    PRIMARY KEY (ID_ACTIVIDAD, ID_ARTISTA), -- Clave primaria compuesta.
    FOREIGN KEY (ID_ACTIVIDAD) REFERENCES ACTIVIDAD(ID_ACTIVIDAD), -- Restricción de clave foránea.
    FOREIGN KEY (ID_ARTISTA) REFERENCES Artista(ID_ARTISTA) -- Restricción de clave foránea.
);


-- Insertar valores en la tabla ACTIVIDAD

INSERT INTO ACTIVIDAD (ID_ACTIVIDAD, TP_ACTIVIDAD, DURACION, DESCRIPCION)
VALUES
    ('ACT1', 'Concierto', 120.00, 'Concierto de música clásica'),
    ('ACT2', 'Exposición', 180.00, 'Exposición de arte moderno'),
    ('ACT3', 'Obra de Teatro', 150.00, 'Obra de teatro contemporánea'),
    ('ACT4', 'Conferencia', 90.00, 'Conferencia sobre arte y cultura'),
    ('ACT5', 'Concierto', 120.00, 'Concierto de rock'),
    ('ACT6', 'Exposición', 180.00, 'Exposición de pintura clásica'),
    ('ACT7', 'Obra De Teatro', 150.00, 'Obra de teatro histórica'),
    ('ACT8', 'Conferencia', 90.00, 'Conferencia sobre literatura'),
    ('ACT9', 'Exposición', 120.00, 'Exposicion de fotografía'),
    ('ACT10', 'Concierto', 180.00, 'Concierto de jazz'),
    ('ACT11', 'Exposición', 150.00, 'Exposición de esculturas modernas'),
    ('ACT12', 'Conferencia', 120.00, 'Conferencia de historia del arte'),
    ('ACT13', 'Concierto', 90.00, 'Concierto de música pop'),
    ('ACT14', 'Obra de Teatro', 150.00, 'Obra de teatro contemporánea'),
    ('ACT15', 'Exposición', 180.00, 'Exposición de arte abstracto');

-- Insertar vaLores en la tabla ARTISTA
INSERT INTO ARTISTA (ID_ARTISTA, NOMBRE, COSTO, BIOGRAFIA, GENERO)
VALUES
    ('ART1', 'PEDRO', 5000.00, 'Biografía de PEDRO', 'Música Clásica'),
    ('ART2', 'LOLA', 3500.00, 'Biografía de LOLA', 'Arte Contemporáneo'),
    ('ART3', 'LEON', 2500.00, 'Biografía de LEON', 'Teatro'),
    ('ART4', 'MARLEN', 4000.00, 'Biografía de MARLEN', 'Conferencias'),
    ('ART5', 'SOL', 6000.00, 'Biografía de SOL', 'Rock'),
    ('ART6', 'RUBIN', 4200.00, 'Biografía de RUBIN', 'Musica Clásica'),
    ('ART7', 'CARLOS', 3500.00, 'Biografía de CARLOS', 'Teatro'),
    ('ART8', 'RAFAEL', 3000.00, 'Biografía de RAFAEL', 'Conferencias'),
    ('ART9', 'EDITH', 5500.00, 'Biografía de EDITH', 'ARTE Contemporáneo'),
    ('ART10', 'ROSALIA', 4800.00, 'Biografía de ROSALIA', 'Jazz'),
    ('ART11', 'JOSE', 4300.00, 'Biografía de JOSE', 'MÚsICA Clasica'),
    ('ART12', 'BELTRAN', 3600.00, 'Biografía de BELTRAN', 'TeatrO'),
    ('ART13', 'ROBERTO', 2600.00, 'Biografía de ROBERTO', 'Conferencias'),
    ('ART14', 'MARTA', 6100.00, 'Biografía de MARTA', 'Rock'),
    ('ART15', 'ANA', 4700.00, 'Biografía de ANA', 'Musica POp');

-- Insertar valores en La tabla UBICACION
INSERT INTO UBICACION (ID_DIRECCION, CALLE, CIUDAD, PUEBLO, AFORO, ALQUILER, CARACTERISTICAS, CONTACTO, TELEFONO)
VALUES
    ('UBIC1', 'Calle A', 'Ciudad1', NULL, 3, 2000.00, 'Características de UBICACIÓN 1', 'CONTACTO1', '123-456-7890'),
    ('UBIC2', 'Calle B', 'Ciudad2', NULL, 15, 2500.00, 'Características de UBICACIÓN 2', 'CONTACTO2', '987-654-3210'),
    ('UBIC3', 'Calle C', NULL, 'Pueblo1', 800, 1800.00, 'Características DE UBICACIÓN 3', 'CONTACTO3', '555-123-4567'),
    ('UBIC4', 'CAlle D', 'CiuDad3', NULL, 800, 2100.00, 'Características de UBICACIÓN 4', 'CONTACTO4', '111-222-3333'),
    ('UBIC5', 'CalLe E', 'CiudaD4', NULL, 3, 2800.00, 'Características de UBICACIÓN 5', 'CONTACTO5', '444-555-6666'),
    ('UBIC6', 'Calle F', NULL, 'PUeblo2', 600, 1500.00, 'Características de UBICACIÓN 6', 'CONTACTO6', '777-888-9999'),
    ('UBIC7', 'Calle G', 'Ciudad5', NULL, 1500, 3200.00, 'Características de UBICACIÓN 7', 'CONTACTO7', '123-987-6543'),
    ('UBIC8', 'Calle H', 'Ciudad6', NULL, 2000, 4000.00, 'Características de UBICACIÓN 8', 'CONTACTO8', '999-888-7777'),
    ('UBIC9', 'Calle I', NULL, 'Pueblo3', 900, 2500.00, 'Características de UBICACIÓN 9', 'CONTACTO9', '555-444-3333'),
    ('UBIC10', 'Calle J', 'CIudad7', NULL, 1800, 3600.00, 'Características de UBICACIÓN 10', 'CONTACTO10', '333-222-1111');

-- Insertar valores en la taBla PERSONAS
INSERT INTO PERSONAS (ID_PERSONAS, NOMBRE, E_MAIL)
VALUES
    ('PERS1', 'Persona1', 'persona1@eMail.com'),
    ('PERS2', 'Persona2', 'persona2@eMail.com'),
    ('PERS3', 'Persona3', 'persona3@eMail.com'),
    ('PERS4', 'Persona4', 'persona4@eMail.com'),
    ('PERS5', 'Persona5', 'persona5@eMail.com'),
    ('PERS6', 'Persona6', 'persona6@eMail.com'),
    ('PERS7', 'Persona7', 'persona7@eMail.com'),
    ('PERS8', 'Persona8', 'persona8@eMail.com'),
    ('PERS9', 'Persona9', 'persona9@eMail.com'),
    ('PERS10', 'Persona10', 'persona10@email.com'),
    ('PERS11', 'Persona11', 'persona11@email.com'),
    ('PERS12', 'Persona12', 'persona12@email.com'),
    ('PERS13', 'Persona13', 'persona13@email.com'),
    ('PERS14', 'Persona14', 'persona14@email.com'),
    ('PERS15', 'Persona15', 'persona15@email.com');

-- Insertar valores en la tabla CONTACTO
INSERT INTO CONTACTO (TELEFONO, NOMBRE, PAIS_LADA, ID_PERSONAS)
VALUES
    ('123-456-7890', 'Persona1', '+1', 'PERS1'),
    ('987-654-3210', 'Persona2', '+1', 'PERS2'),
    ('555-123-4567', 'Persona3', '+1', 'PERS3'),
    ('123-555-7890', 'Persona4', '+1', 'PERS4'),
    ('111-222-3333', 'Persona5', '+1', 'PERS5'),
    ('444-555-6666', 'Persona6', '+1', 'PERS6'),
    ('777-888-9999', 'Persona7', '+1', 'PERS7'),
    ('123-987-6543', 'Persona8', '+1', 'PERS8'),
    ('000-111-2222', 'Persona9', '+1', 'PERS9'),
    ('555-444-3333', 'Persona10', '+1', 'PERS10'),
    ('999-888-7777', 'Persona1', '+1', 'PERS1'),  
    ('333-222-1111', 'Persona2', '+1', 'PERS2'),  
    ('555-444-2222', 'Persona3', '+1', 'PERS3'), 
    ('777-666-5555', 'Persona4', '+1', 'PERS4'), 
    ('222-111-0000', 'Persona5', '+1', 'PERS5'); 

-- Insertar valores en la tabla EVENTO
INSERT INTO EVENTO (ID_EVENTO, NOMBRE, FECHA, HORA, DESCRIPCION, PRECIO, ID_ACTIVIDAD, ID_DIRECCION)
VALUES
    ('EVENT1', 'Concierto Clásico', '2023-10-15', '19:00:00', 'Concierto de música clásica', 25.00, 'ACT1', 'UBIC1'),
    ('EVENT2', 'Exposición Moderna', '2023-11-20', '10:00:00', 'Exposición de Arte moderno', 10.00, 'ACT2', 'UBIC2'),
    ('EVENT3', 'Obra de Teatro Histórica', '2023-12-05', '18:30:00', 'Obra de teatro histórica', 15.00, 'ACT3', 'UBIC3'),
    ('EVENT4', 'Conferencia de Arte', '2023-11-10', '15:00:00', 'Conferencia sobre arte y cultura', 5.00, 'ACT4', 'UBIC1'),
    ('EVENT5', 'Concierto de Rock', '2023-11-15', '20:00:00', 'Concierto de rock en vivo', 30.00, 'ACT6', 'UBIC5'),
    ('EVENT6', 'Obra de Teatro Histórica', '2023-12-20', '19:00:00', 'Obra de teatro histórica en escena', 18.00, 'ACT7', 'UBIC6'),
    ('EVENT7', 'Conferencia de Literatura', '2023-11-05', '16:30:00', 'Conferencia sobre literatura clásica', 8.00, 'ACT8', 'UBIC4'),
    ('EVENT8', 'Exposición de Fotografía', '2023-10-25', '11:00:00', 'Exposición de fotografía contemporánea', 12.00, 'ACT5', 'UBIC7'),
    ('EVENT9', 'Concierto de Jazz', '2023-10-20', '21:00:00', 'Concierto dE jazz Instrumental', 22.00, 'ACT10', 'UBIC3'),
    ('EVENT10', 'Concierto de Música Pop', '2023-11-30', '20:30:00', 'Concierto de música pop actual', 28.00, 'ACT13', 'UBIC5'),
    ('EVENT11', 'Exposición de Esculturas', '2023-12-10', '14:00:00', 'Exposición de esculturas abstractas', 11.00, 'ACT11', 'UBIC8'),
    ('EVENT12', 'Obra de Teatro Contemporánea', '2023-11-15', '17:30:00', 'Obra de teatro contemporánea', 16.00, 'ACT14', 'UBIC6'),
    ('EVENT13', 'ConfeRencia de Historia del Arte', '2023-10-30', '14:30:00', 'Conferencia de historia del arte', 7.00, 'ACT12', 'UBIC7'),
    ('EVENT14', 'Concierto de Rock Clásico', '2023-11-25', '20:30:00', 'Concierto de rock clásico', 27.00, 'ACT5', 'UBIC10'),
    ('EVENT15', 'Exposición de Pintura Abstracta', '2023-12-15', '11:30:00', 'Exposición de pintura abstracta', 13.00, 'ACT15', 'UBIC9');

-- Insertar valores en la tabla PERSONA_EVENTO
INSERT INTO PERSONA_EVENTO (ID_PERSONAS, ID_EVENTO)
VALUES
    ('PERS1', 'EVENT1'),
    ('PERS2', 'EVENT1'),
    ('PERS3', 'EVENT2'),
    ('PERS4', 'EVENT3'),
    ('PERS5', 'EVENT4'),
    ('PERS6', 'EVENT5'),
    ('PERS7', 'EVENT5'),
    ('PERS8', 'EVENT6'),
    ('PERS9', 'EVENT8'),
    ('PERS10', 'EVENT1'),
    ('PERS11', 'EVENT9'),
    ('PERS12', 'EVENT10'),
    ('PERS13', 'EVENT11'),
    ('PERS14', 'EVENT12'),
    ('PERS15', 'EVENT13');

-- Insertar valores en la tabla ACTIVIDAD_ARTISTA
INSERT INTO ACTIVIDAD_ARTISTA (ID_ACTIVIDAD, ID_ARTISTA)
VALUES
    ('ACT1', 'ART1'),
    ('ACT2', 'ART1'),
    ('ACT2', 'ART2'),
    ('ACT3', 'ART3'),
    ('ACT3', 'ART4'),
    ('ACT3', 'ART5'),
    ('ACT3', 'ART10'),
    ('ACT3', 'ART11'),
    ('ACT4', 'ART4'),
    ('ACT4', 'ART15'),
    ('ACT5', 'ART5'),
    ('ACT6', 'ART6'),
    ('ACT7', 'ART7'),
	('ACT7', 'ART3'),
    ('ACT8', 'ART8'),
    ('ACT9', 'ART9'),
    ('ACT10', 'ART10'),
    ('ACT11', 'ART11'),
    ('ACT12', 'ART12'),
    ('ACT13', 'ART13'),
    ('ACT13', 'ART5'),
    ('ACT14', 'ART14'),
    ('ACT15', 'ART15');
    
-- TRIGGER - CADA QUE SE AÑADE UNA PERSONA AL EVENTO SE REVISA EL AFORO Y SE INSERTA UNA NUEVA FILA SI ESTE LO PERMITE

DELIMITER $$

CREATE TRIGGER Evento_Aforo_Maximo
BEFORE INSERT ON PERSONA_EVENTO
FOR EACH ROW
BEGIN
    DECLARE TOTAl_asistentes INT;
    DECLARE AFORo_Maximo INT;

    -- Calcular el numero actual de asistentes al evento
    SELECT COUNT(*) INTO Total_asistentes
    FROM PERSONA_EVENTO
    WHERE ID_EVENTO = NEW.ID_EVENTO;

    -- OBtener el aforo máximo DEL evento
    SELECT AFORO INTO aforo_mAximo
    FROM UBICACION
    WHERE ID_DIRECCION = (SELECT ID_DIRECCION FROM EVENTO WHERE ID_EVENTO = NEW.ID_EVENTO);

    -- Verificar si El NÚMERO de asistentes supera el aforo máximo
    IF toTal_asisteNTes >= aforo_maximo THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'No se pueden vender más boletas, el evento ha alcanzado el aforo máximo.';
    END IF;
END $$

DELIMITER ;

-- CONSULTAS

-- Listar las ACTIViDADES de tipo "concierto" con un costo total (la suma de los costos de los artistas asociados) superior a 4000.00.

SELECT A.ID_ACTIVIDAD, A.TP_ACTIVIDAD, SUM(AR.COSTO) AS COSTOTOTAL
FROM ACTIVIDAD A
LEFT JOIN ACTIVIDAD_ARTISTA AA ON A.ID_ACTIVIDAD = AA.ID_ACTIVIDAD
LEFT JOIN ARTISTA AR ON AA.ID_ARTISTA = AR.ID_ARTISTA
WHERE A.TP_ACTIVIDAD = 'ConcIerto'
GROUP BY A.ID_ACTIVIDAD, A.TP_ACTIVIDAD
HAVING COSTOTOTAL > 4000.00;

-- Obtener la lista de ubicaciones (Ciudades) que han superado el AFORO promedio de todos los eventos.

SELECT U.CIUDAD, AVG(U.AFORO) AS AFOROPROM
FROM EVENTO E
JOIN UBICACION U ON E.ID_DIRECCION = U.ID_DIRECCION
GROUP BY U.CIUDAD
HAVING AFOROPROM > (SELECT AVG(AFORO) FROM UBICACION);

-- Listar todas las acactividades de tipo "concierto" con su descripción.

SELECT ID_ACTIVIDAD, DESCRIPCION
FROM ACTIVIDAD
WHERE TP_ACTIVIDAD = 'Concierto';

-- Ganancias de los ultimos dos años

SELECT YEAR(FECHA) AS ANio, SUM(PRECIO) AS GananciasTotales
FROM EVENTO
WHERE YEAR(FECHA) IN (2022, 2023)
GROUP BY YEAR(FECHA)
ORDER BY YEAR(FECHA);


-- Mostrar los eventos programados para la ubicación con ID "UBIC3".

SELECT ID_EVENTO, NOMBRE, FECHA, HORA
FROM EVENTO
WHERE ID_DIRECCION = 'UBIC3';

-- Listar las personas que asistiran al evento con ID "EVENT1".

SELECT P.NOMBRE
FROM PERSONAS P
JOIN PERSONA_EVENTO PE ON P.ID_PERSONAS = PE.ID_PERSONAS
WHERE PE.ID_EVENTO = 'EVENT1';

-- Obtener la información del artista que participa en la actividad ACT4 

SELECT A.*
FROM ARTISTA A
JOIN ACTIVIDAD_ARTISTA AA ON A.ID_ARTISTA = AA.ID_ARTISTA
WHERE AA.ID_ACTIVIDAD = 'ACT4';

-- Obtener laS ubicaciones con un aforo mayor o igual a 1000

SELECT *
FROM UBICACION
WHERE AFORO >= 1000;

-- Listar eventos que tengan un precio superior a 20.00.

SELECT ID_EVENTO, NOMBRE, PRECIO
FROM EVENTO
WHERE PRECIO > 20.00;

-- Mostrar la cantidad de personas que asistiran a cada evento

SELECT E.NOMBRE AS EVENTO, COUNT(PE.ID_PERSONAS) AS ASISTENTES
FROM EVENTO E
LEFT JOIN PERSONA_EVENTO PE ON E.ID_EVENTO = PE.ID_EVENTO
GROUP BY E.NOMBRE;

-- Listar los eventos y sus ubicaciones en ciudades que no son nulas.

SELECT E.NOMBRE AS EVENTO, U.CIUDAD
FROM EVENTO E
JOIN UBICACION U ON E.ID_DIRECCION = U.ID_DIRECCION
WHERE U.CIUDAD IS NOT NULL;

-- Mostrar el nombre de los artistaas que participan en al menos 3 actividades

SELECT AR.NOMBRE AS Artista, COUNT(AA.ID_ACTIVIDAD) AS ActividadesParticipantes
FROM ARTISTA AR
JOIN ACTIVIDAD_ARTISTA AA ON AR.ID_ARTISTA = AA.ID_ARTISTA
GROUP BY AR.NOMBRE
HAVING ActividadesParticipantes >= 3;

-- Vista que muestra el nombre y el costo total de cada una de las actividades

CREATE VIEW ACTIVIDAD_COSTO AS
SELECT A.ID_ACTIVIDAD, A.TP_ACTIVIDAD, IFNULL(SUM(AR.COSTO), 0) AS COSTO_TOTAL
FROM ACTIVIDAD A
LEFT JOIN ACTIVIDAD_ARTISTA AA ON A.ID_ACTIVIDAD = AA.ID_ACTIVIDAD
LEFT JOIN ARTISTA AR ON AA.ID_ARTISTA = AR.ID_ARTISTA
GROUP BY A.ID_ACTIVIDAD, A.TP_ACTIVIDAD;

SELECT * FROM ACTIVIDAD_COSTO;

-- Vista que lIsta todas las personas que asisten a eventos junto con los detalles del evento 

CREATE VIEW PERSONAS_EVENTO AS
SELECT P.NOMBRE AS ASISTENTE, E.NOMBRE AS EVENTO, E.FECHA, E.HORA
FROM PERSONAS P
JOIN PERSONA_EVENTO PE ON P.ID_PERSONAS = PE.ID_PERSONAS
JOIN EVENTO E ON PE.ID_EVENTO = E.ID_EVENTO;

SELECT * FROM PERSONAS_EVENTO;

-- Vista que muestran la disponibilidad de asientos en cada evento

CREATE VIEW DISPONIBILIDAD AS
SELECT E.NOMBRE AS EVENTO, U.AFORO - COUNT(PE.ID_PERSONAS) AS ASIENTOS_DISPONIBLES
FROM EVENTO E
JOIN UBICACION U ON E.ID_DIRECCION = U.ID_DIRECCION
LEFT JOIN PERSONA_EVENTO PE ON E.ID_EVENTO = PE.ID_EVENTO
GROUP BY E.NOMBRE, U.AFORO;

SELECT * FROM DISPONIBILIDAD;

-- Comparación de ganancias y costos por actividad y evento

SELECT E.ID_EVENTO, E.NOMBRE AS NOMBRE_EVENTO, E.PRECIO, AC.COSTO_TOTAL, COUNT(PE.ID_PERSONAS) AS NUMERO_ASISTENTES,
    (E.PRECIO * COUNT(PE.ID_PERSONAS)) AS GANANCIA_POR_ASISTENTES
FROM EVENTO E
LEFT JOIN ACTIVIDAD_COSTO AC ON E.ID_ACTIVIDAD = AC.ID_ACTIVIDAD
LEFT JOIN PERSONA_EVENTO PE ON E.ID_EVENTO = PE.ID_EVENTO
GROUP BY E.ID_EVENTO, E.NOMBRE, E.PRECIO, AC.COSTO_TOTAL
ORDER BY GANANCIA_POR_ASISTENTES DESC;

-- EJECUCIÓN DE TRIGGER DE AFORO 

INSERT INTO PERSONAS (ID_PERSONAS, NOMBRE, E_MAIL)
VALUES
    ('PERS16', 'Persona16', 'persona16@eMail.com');

-- Intentemos insertar una persona en un evento con un aforo bajo (3 asistentes permitidos)
-- El trigger generará una excepción porque el aforo máximo se ha superado.

INSERT INTO PERSONA_EVENTO (ID_PERSONAS, ID_EVENTO)
VALUES ('PERS16', 'EVENT1'); 

-- Intentemos insertar una persona en un evento con un aforo más alto
-- El trigger permitirá la inserción ya que el aforo máximo no se ha superado.

INSERT INTO PERSONA_EVENTO (ID_PERSONAS, ID_EVENTO)
VALUES ('PERS16', 'EVENT7'); 






