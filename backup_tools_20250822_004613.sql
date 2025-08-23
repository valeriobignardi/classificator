-- MySQL dump 10.13  Distrib 8.0.43, for Linux (x86_64)
--
-- Host: localhost    Database: TAG
-- ------------------------------------------------------
-- Server version	8.0.43-0ubuntu0.24.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `tools`
--

DROP TABLE IF EXISTS `tools`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tools` (
  `id` int NOT NULL AUTO_INCREMENT,
  `tool_name` varchar(100) NOT NULL,
  `display_name` varchar(150) NOT NULL,
  `description` text NOT NULL,
  `function_schema` json NOT NULL,
  `is_active` tinyint(1) DEFAULT '1',
  `tenant_id` varchar(36) DEFAULT NULL,
  `tenant_name` varchar(100) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `tool_name` (`tool_name`),
  KEY `idx_tool_name` (`tool_name`),
  KEY `idx_tenant` (`tenant_id`,`tenant_name`),
  KEY `idx_active` (`is_active`),
  KEY `idx_created` (`created_at`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `tools`
--

LOCK TABLES `tools` WRITE;
/*!40000 ALTER TABLE `tools` DISABLE KEYS */;
INSERT INTO `tools` VALUES (1,'_get_available_labels','Recupera Etichette Disponibili','Recupera tutte le etichette disponibili dal database TAG per il tenant specifico','{\"type\": \"function\", \"function\": {\"name\": \"_get_available_labels\", \"parameters\": {\"type\": \"object\", \"required\": [\"tenant_name\"], \"properties\": {\"tenant_name\": {\"type\": \"string\", \"description\": \"Nome del tenant\"}}}, \"description\": \"Recupera etichette disponibili dal database TAG\"}}',1,'015007d9-d413-11ef-86a5-96000228e7fe','Humanitas','2025-08-22 00:28:08','2025-08-22 00:28:08'),(2,'_get_priority_labels_hint','Suggerimenti Etichette Prioritarie','Fornisce suggerimenti sulle etichette più frequenti per prioritizzazione nella classificazione','{\"type\": \"function\", \"function\": {\"name\": \"_get_priority_labels_hint\", \"parameters\": {\"type\": \"object\", \"required\": [\"tenant_name\"], \"properties\": {\"tenant_name\": {\"type\": \"string\", \"description\": \"Nome del tenant\"}}}, \"description\": \"Suggerisce etichette prioritarie basate sulla frequenza\"}}',1,'015007d9-d413-11ef-86a5-96000228e7fe','Humanitas','2025-08-22 00:28:08','2025-08-22 00:28:08'),(3,'_get_dynamic_examples','Esempi Dinamici per Training','Recupera esempi dinamici selezionati semanticamente per il training del classificatore','{\"type\": \"function\", \"function\": {\"name\": \"_get_dynamic_examples\", \"parameters\": {\"type\": \"object\", \"required\": [\"conversation_text\"], \"properties\": {\"max_examples\": {\"type\": \"integer\", \"default\": 4, \"description\": \"Numero massimo di esempi da restituire\"}, \"conversation_text\": {\"type\": \"string\", \"description\": \"Testo della conversazione\"}}}, \"description\": \"Recupera esempi dinamici per training\"}}',1,'015007d9-d413-11ef-86a5-96000228e7fe','Humanitas','2025-08-22 00:28:08','2025-08-22 00:28:08'),(4,'_summarize_if_long','Riassunto Conversazioni Lunghe','Riassume il testo della conversazione se supera la lunghezza massima specificata','{\"type\": \"function\", \"function\": {\"name\": \"_summarize_if_long\", \"parameters\": {\"type\": \"object\", \"required\": [\"conversation_text\"], \"properties\": {\"max_length\": {\"type\": \"integer\", \"default\": 2000, \"description\": \"Lunghezza massima prima del riassunto\"}, \"conversation_text\": {\"type\": \"string\", \"description\": \"Testo della conversazione\"}}}, \"description\": \"Riassume conversazioni troppo lunghe\"}}',1,'015007d9-d413-11ef-86a5-96000228e7fe','Humanitas','2025-08-22 00:28:08','2025-08-22 00:28:08');
/*!40000 ALTER TABLE `tools` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-08-22  0:46:13
