const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { v4: uuidv4 } = require('uuid');
const path = require('path');

const { readUsers, addUser, findUserByEmail, hash } = require(path.resolve(__dirname, '..', 'src', 'lib', 'userStore'));

const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 3001;

// Simple in-memory sessions: token -> email
const sessions = {};

app.post('/auth/signup', async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!email || !password) return res.status(400).json({ message: 'Missing email or password' });
    try {
      const user = await addUser(email, password);
      const token = uuidv4();
      sessions[token] = user.email;
      return res.json({ user: { id: user.email, email: user.email }, token });
    } catch (err) {
      return res.status(400).json({ message: err.message });
    }
  } catch (err) {
    return res.status(500).json({ message: 'Server error' });
  }
});

app.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ message: 'Missing email or password' });
    const user = await findUserByEmail(email);
    if (!user) return res.status(400).json({ message: 'Invalid credentials' });
    if (user.passwordHash !== hash(password)) return res.status(400).json({ message: 'Invalid credentials' });
    const token = uuidv4();
    sessions[token] = user.email;
    return res.json({ user: { id: user.email, email: user.email }, token });
  } catch (err) {
    return res.status(500).json({ message: 'Server error' });
  }
});

app.post('/auth/logout', (req, res) => {
  try {
    const auth = req.headers.authorization || '';
    const token = auth.replace(/^Bearer\s+/i, '') || req.body?.token;
    if (token && sessions[token]) delete sessions[token];
    return res.json({ ok: true });
  } catch (err) {
    return res.status(500).json({ message: 'Server error' });
  }
});

app.get('/auth/session', async (req, res) => {
  try {
    const auth = req.headers.authorization || '';
    const token = auth.replace(/^Bearer\s+/i, '');
    const email = sessions[token];
    if (!email) return res.status(401).json({ message: 'No session' });
    const user = await findUserByEmail(email);
    return res.json({ user: { id: user.email, email: user.email } });
  } catch (err) {
    return res.status(500).json({ message: 'Server error' });
  }
});

app.get('/auth/profile', async (req, res) => {
  try {
    const auth = req.headers.authorization || '';
    const token = auth.replace(/^Bearer\s+/i, '');
    const email = sessions[token];
    if (!email) return res.status(401).json({ message: 'No session' });
    const user = await findUserByEmail(email);
    if (!user) return res.status(404).json({ message: 'User not found' });
    return res.json({ id: user.email, email: user.email, created_at: user.createdAt || null });
  } catch (err) {
    return res.status(500).json({ message: 'Server error' });
  }
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Auth server listening on http://localhost:${PORT}`);
});
