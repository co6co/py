/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : systemdb

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-09-27 12:00:46
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_menu_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_menu_role`;
CREATE TABLE `sys_menu_role` (
  `menu_id` int(11) NOT NULL COMMENT '主键id',
  `role_id` bigint(20) NOT NULL COMMENT '主键id',
  `create_user` bigint(20) DEFAULT NULL COMMENT '创建人',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`menu_id`,`role_id`),
  KEY `role_id` (`role_id`),
  CONSTRAINT `sys_menu_role_ibfk_1` FOREIGN KEY (`menu_id`) REFERENCES `sys_menu` (`id`) ON DELETE CASCADE,
  CONSTRAINT `sys_menu_role_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_menu_role
-- ----------------------------
INSERT INTO `sys_menu_role` VALUES ('3', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('4', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('5', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('6', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('7', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('8', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('9', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('20', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('21', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('22', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('23', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('25', '1', '1', '2024-09-27 12:00:11');
INSERT INTO `sys_menu_role` VALUES ('30', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('31', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('32', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('33', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('34', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('35', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('36', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('37', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('38', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('39', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('40', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('50', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('51', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('52', '1', '1', '2024-09-27 12:00:11');
INSERT INTO `sys_menu_role` VALUES ('53', '1', '1', '2024-09-27 12:00:11');
INSERT INTO `sys_menu_role` VALUES ('61', '1', '1', '2024-09-27 12:00:11');
INSERT INTO `sys_menu_role` VALUES ('62', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('63', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('64', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('65', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('66', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('67', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('68', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('69', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('70', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('71', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('72', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('73', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('74', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('75', '1', '1', '2024-09-27 12:00:19');
INSERT INTO `sys_menu_role` VALUES ('76', '1', '1', '2024-09-27 12:00:19');
