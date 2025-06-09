import express from 'express';
import { handleChatbotQuery } from '../controllers/chatbotController.js';
const router = express.Router();

router.post('/', handleChatbotQuery);
export default router;
