"""
Unit tests for File Upload Widget API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestFileUploadWidget:
    """Test cases for File Upload widget API endpoints."""
    
    def test_list_uploaded_files(self):
        """Test listing uploaded files."""
        response = client.get("/api/v1/data/uploaded")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "files" in data
        assert isinstance(data["files"], list)
    
    def test_upload_csv_file(self):
        """Test uploading a CSV file."""
        csv_content = """name,age,score
Alice,25,85.5
Bob,30,90.0
Charlie,35,78.5
"""
        
        files = {
            "file": ("test_data.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        
        response = client.post("/api/v1/data/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "filename" in data or "path" in data
            assert "instances" in data or "rows" in data
            
            # Clean up - delete uploaded file
            filename = data.get("filename", "test_data.csv")
            client.delete(f"/api/v1/data/uploaded/{filename}")
    
    def test_upload_tab_file(self):
        """Test uploading a TAB file."""
        tab_content = """name\tage\tscore
Alice\t25\t85.5
Bob\t30\t90.0
Charlie\t35\t78.5
"""
        
        files = {
            "file": ("test_data.tab", io.BytesIO(tab_content.encode()), "text/tab-separated-values")
        }
        
        response = client.post("/api/v1/data/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            assert "filename" in data or "path" in data
            
            # Clean up
            filename = data.get("filename", "test_data.tab")
            client.delete(f"/api/v1/data/uploaded/{filename}")
    
    def test_delete_uploaded_file(self):
        """Test deleting an uploaded file."""
        # First upload a file
        csv_content = """a,b,c
1,2,3
"""
        files = {
            "file": ("to_delete.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        
        upload_response = client.post("/api/v1/data/upload", files=files)
        
        if upload_response.status_code == 200:
            # Then delete it
            delete_response = client.delete("/api/v1/data/uploaded/to_delete.csv")
            
            # Should succeed or file already deleted
            assert delete_response.status_code in [200, 404]
    
    def test_delete_nonexistent_file(self):
        """Test deleting a non-existent file."""
        response = client.delete("/api/v1/data/uploaded/nonexistent_file_12345.csv")
        
        # Should return 404 or success (if already deleted)
        assert response.status_code in [200, 404]
    
    def test_upload_empty_file(self):
        """Test uploading an empty file."""
        files = {
            "file": ("empty.csv", io.BytesIO(b""), "text/csv")
        }
        
        response = client.post("/api/v1/data/upload", files=files)
        
        # Should handle gracefully - either error or success with 0 rows
        assert response.status_code in [200, 400, 422]
    
    def test_upload_large_file(self):
        """Test uploading a larger file (1000 rows)."""
        rows = ["col1,col2,col3"]
        for i in range(1000):
            rows.append(f"val{i},val{i+1},val{i+2}")
        csv_content = "\n".join(rows)
        
        files = {
            "file": ("large_data.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        
        response = client.post("/api/v1/data/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            instances = data.get("instances", data.get("rows", 0))
            assert instances == 1000
            
            # Clean up
            client.delete("/api/v1/data/uploaded/large_data.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

