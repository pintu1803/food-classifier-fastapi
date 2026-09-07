from fastapi import HTTPException, UploadFile

#Allowed image types coming as in input
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

#Max size allowed for image # 5 MB
MAX_IMAGE_SIZE = 5 * 1024 * 1025

#Validation of input image
async def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPEG and WEBP images are allowed"
        )

#Validation of input size
async def validate_size(file):
    content = await file.read()

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeded 5MB"
        )
    
    await file.seek(0)