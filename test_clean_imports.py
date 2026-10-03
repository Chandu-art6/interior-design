import sys

print("=== Checking Module Imports for Circular Dependencies ===")

try:
    print("1. Importing app.schemas.design...")
    import app.schemas.design as design_schemas
    print("   [OK] app.schemas.design imported")

    print("2. Importing config...")
    import config
    print("   [OK] config imported")

    print("3. Importing database.models and database.session...")
    import database.models as db_models
    import database.session as db_session
    print("   [OK] database layer imported")

    print("4. Importing services.recommender...")
    from services.recommender import recommender_service, DesignRecommender
    print("   [OK] recommender_service imported:", type(recommender_service))

    print("5. Importing services.controlnet_service...")
    from services.controlnet_service import controlnet_service
    print("   [OK] controlnet_service imported")

    print("6. Importing services.yolo_service...")
    from services.yolo_service import yolo_service
    print("   [OK] yolo_service imported")

    print("7. Importing ui.components.widgets and ui.widgets...")
    import ui.components.widgets as widgets_comp
    import ui.widgets as widgets_proxy
    print("   [OK] ui.widgets imported")

    print("8. Importing ui.api_client...")
    from ui.api_client import api_client
    print("   [OK] api_client imported")

    print("9. Importing app.api.routes...")
    from app.api.routes import router
    print("   [OK] app.api.routes imported")

    print("\n[ALL PASSED] Zero circular imports or dependency errors detected across all services and UI layers!")

except Exception as e:
    print(f"\n[FAILED] Import Error Detected: {e}")
    sys.exit(1)
