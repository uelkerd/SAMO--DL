"""
Prisma client utility for the SAMO-DL application.

This module provides functions to interact with the Prisma client via subprocess calls.
It's a simple wrapper that allows Python code to execute Prisma commands.
"""

import os
import json
import subprocess
from typing import Dict, List, Any, Optional


class PrismaClient:
    """
    A simple wrapper class for Prisma client operations.
    
    This class allows executing Prisma operations from Python by running Node.js scripts.
    """
    
    @staticmethod
    def execute_prisma_command(script: str) -> Dict[str, Any]:
        """
        Execute a Node.js script that uses Prisma client.
        
        Args:
            script (str): The JavaScript code to execute
            
        Returns:
            Dict[str, Any]: The result of the operation as a dictionary
        
        Raises:
            Exception: If the script execution fails
        """
        # Create a temporary JS file
        with open('temp_prisma_script.js', 'w') as f:
            f.write(f"""
const {{ PrismaClient }} = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {{
    try {{
        const result = await (async () => {{
            {script}
        }})();
        console.log(JSON.stringify(result));
        await prisma.$disconnect();
        return result;
    }} catch (e) {{
        console.error(e);
        await prisma.$disconnect();
        process.exit(1);
    }}
}}

main();
""")
        
        try:
            # Execute the script
            result = subprocess.run(['node', 'temp_prisma_script.js'], 
                                   capture_output=True, 
                                   text=True, 
                                   check=True)
            
            # Parse the output
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Prisma command failed: {e.stderr}")
        finally:
            # Clean up the temporary file
            if os.path.exists('temp_prisma_script.js'):
                os.remove('temp_prisma_script.js')
    
    def create_user(self, email: str, password_hash: str, consent_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            email (str): User's email
            password_hash (str): Hashed password
            consent_version (str, optional): Version of consent the user agreed to
            
        Returns:
            Dict[str, Any]: Created user data
        """
        script = f"""
        return prisma.user.create({{
            data: {{
                email: '{email}',
                passwordHash: '{password_hash}',
                consentVersion: {f"'{consent_version}'" if consent_version else 'null'},
                consentGivenAt: {f"new Date()" if consent_version else 'null'}
            }}
        }});
        """
        
        return self.execute_prisma_command(script)
    
    def create_journal_entry(self, user_id: str, title: str, content: str, is_private: bool = True) -> Dict[str, Any]:
        """
        Create a new journal entry.
        
        Args:
            user_id (str): ID of the user who owns this entry
            title (str): Entry title
            content (str): Entry content
            is_private (bool): Whether the entry is private
            
        Returns:
            Dict[str, Any]: Created journal entry data
        """
        script = f"""
        return prisma.journalEntry.create({{
            data: {{
                userId: '{user_id}',
                title: '{title}',
                content: '{content}',
                isPrivate: {str(is_private).lower()},
                user: {{
                    connect: {{
                        id: '{user_id}'
                    }}
                }}
            }}
        }});
        """
        
        return self.execute_prisma_command(script)
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get a user by email.
        
        Args:
            email (str): Email to lookup
            
        Returns:
            Optional[Dict[str, Any]]: User data or None if not found
        """
        script = f"""
        return prisma.user.findUnique({{
            where: {{ email: '{email}' }}
        }});
        """
        
        result = self.execute_prisma_command(script)
        return result if result else None
    
    def get_journal_entries_by_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get journal entries for a specific user.
        
        Args:
            user_id (str): User ID
            limit (int): Maximum number of entries to return
            
        Returns:
            List[Dict[str, Any]]: List of journal entries
        """
        script = f"""
        return prisma.journalEntry.findMany({{
            where: {{ userId: '{user_id}' }},
            take: {limit},
            orderBy: {{ createdAt: 'desc' }}
        }});
        """
        
        return self.execute_prisma_command(script) 