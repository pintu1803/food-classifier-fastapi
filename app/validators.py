from fastapi import HTTPException, UploadFile

#Allowed image types coming as in input
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

#Max size allowed for image # 5 MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024

#Validation of input image
#does not need bytes or PIL, can read from metadata of uploaded file
async def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPEG and WEBP images are allowed"
        )

#Validation of input size
async def validate_size(content: bytes):
    #passed the total bytes 
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeded 5MB"
        )
    