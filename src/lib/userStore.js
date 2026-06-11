const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

const usersFile = path.resolve(process.cwd(), 'data', 'users.json');

async function readUsers() {
  try {
    const data = await fs.readFile(usersFile, 'utf8');
    return JSON.parse(data || '[]');
  } catch (err) {
    if (err.code === 'ENOENT') {
      await fs.mkdir(path.dirname(usersFile), { recursive: true });
      await fs.writeFile(usersFile, '[]', 'utf8');
      return [];
    }
    throw err;
  }
}

function hash(password) {
  return crypto.createHash('sha256').update(String(password)).digest('hex');
}

async function addUser(email, password) {
  const users = await readUsers();
  if (users.find((u) => u.email === email)) throw new Error('User already exists');
  const user = {
    email,
    passwordHash: hash(password),
    createdAt: new Date().toISOString(),
  };
  users.push(user);
  await fs.writeFile(usersFile, JSON.stringify(users, null, 2), 'utf8');
  return user;
}

async function findUserByEmail(email) {
  const users = await readUsers();
  return users.find((u) => u.email === email) || null;
}

module.exports = { readUsers, addUser, findUserByEmail, hash };
