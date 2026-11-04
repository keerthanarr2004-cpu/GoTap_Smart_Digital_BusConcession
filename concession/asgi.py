import os
import django
from django.core.asgi import get_asgi_application
import threading
import asyncio

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'concession.settings')

# Initialize Django
django.setup()

# Start the background task
def start_background_task():
    from stc.tasks import delete_expired_passes_task
    asyncio.run(delete_expired_passes_task())

# Run the task in a separate thread
threading.Thread(target=start_background_task, daemon=True).start()

# Get the ASGI application
application = get_asgi_application()
